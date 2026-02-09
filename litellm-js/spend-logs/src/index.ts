import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import { PrismaClient } from '@prisma/client'
import {LiteLLM_SpendLogs, LiteLLM_IncrementSpend, LiteLLM_IncrementObject} from './_types'

const app = new Hono()

/**
 * Configure PrismaClient with explicit connection pool settings to prevent
 * connection pool exhaustion under high load (fixes #20427).
 *
 * - connection_limit: cap per-instance pool size so that multiple replicas
 *   (e.g. 2 replicas × 2 workers) don't collectively exceed PostgreSQL's
 *   max_connections.
 * - pool_timeout: fail fast when all pooled connections are busy instead of
 *   hanging indefinitely, which would stall the flush loop.
 */
const databaseUrl = process.env.DATABASE_URL || ''
const prisma = new PrismaClient({
  datasources: {
    client: {
      url: appendPoolParams(databaseUrl, {
        connection_limit: '5',
        pool_timeout: '10',
      }),
    },
  },
  log: [
    { level: 'warn', emit: 'stdout' },
    { level: 'error', emit: 'stdout' },
  ],
})

/**
 * Append connection-pool query parameters to a database URL.
 * Handles URLs that already contain query strings.
 */
function appendPoolParams(url: string, params: Record<string, string>): string {
  if (!url) return url
  const separator = url.includes('?') ? '&' : '?'
  const qs = Object.entries(params)
    .map(([k, v]) => `${k}=${v}`)
    .join('&')
  return `${url}${separator}${qs}`
}

// In-memory storage for logs
let spend_logs: LiteLLM_SpendLogs[] = [];
const key_logs: LiteLLM_IncrementObject[] = [];
const user_logs: LiteLLM_IncrementObject[] = [];
const transaction_logs: LiteLLM_IncrementObject[] = [];

const MIN_LOGS = 1; // Minimum number of logs needed to initiate a flush
const FLUSH_INTERVAL = 5000; // Time in ms to wait before trying to flush again
const BATCH_SIZE = 100; // Preferred size of each batch to write to the database
const MAX_LOGS_PER_INTERVAL = 1000; // Maximum number of logs to flush in a single interval
const MAX_QUEUE_SIZE = 10_000; // Upper bound on in-memory queue to prevent unbounded growth
const MAX_RETRY_COUNT = 3; // Maximum retries per batch before dropping it
const INITIAL_RETRY_DELAY_MS = 500; // Starting delay for exponential backoff

/** Track whether the DB connection is healthy for the /health endpoint */
let isDbHealthy = true;
/** Track whether a flush is already in progress to prevent overlapping flushes */
let isFlushInProgress = false;

/**
 * Sleep helper for retry backoff.
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

/**
 * Write a single batch to the database with retry + exponential backoff.
 * Returns true if the batch was written successfully, false otherwise.
 */
async function writeBatchWithRetry(batch: LiteLLM_SpendLogs[]): Promise<boolean> {
  const batchWithDates = batch.map(entry => ({
    ...entry,
    startTime: new Date(entry.startTime),
    endTime: new Date(entry.endTime),
  }));

  for (let attempt = 1; attempt <= MAX_RETRY_COUNT; attempt++) {
    try {
      await prisma.liteLLM_SpendLogs.createMany({
        data: batchWithDates,
      });
      isDbHealthy = true;
      return true;
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : String(error);
      console.error(
        `[flush] Batch write failed (attempt ${attempt}/${MAX_RETRY_COUNT}): ${errMsg}`
      );

      if (attempt < MAX_RETRY_COUNT) {
        const delay = INITIAL_RETRY_DELAY_MS * Math.pow(2, attempt - 1);
        await sleep(delay);
      }
    }
  }

  // All retries exhausted
  isDbHealthy = false;
  return false;
}

const flushLogsToDb = async () => {
  if (isFlushInProgress) {
    console.log('[flush] Previous flush still in progress, skipping this cycle.');
    return;
  }

  if (spend_logs.length < MIN_LOGS) {
    if (spend_logs.length > 0) {
      console.log(`Accumulating logs. Currently at ${spend_logs.length}, waiting for at least ${MIN_LOGS}.`);
    } else {
      console.log("No logs to flush.");
    }
    return;
  }

  isFlushInProgress = true;

  try {
    // Limit the logs to process in this interval to MAX_LOGS_PER_INTERVAL or less
    const logsToProcess = spend_logs.slice(0, MAX_LOGS_PER_INTERVAL);
    // Remove them from the queue upfront; failed batches will be re-queued
    spend_logs = spend_logs.slice(logsToProcess.length);

    const failedLogs: LiteLLM_SpendLogs[] = [];

    for (let i = 0; i < logsToProcess.length; i += BATCH_SIZE) {
      const batch = logsToProcess.slice(i, i + BATCH_SIZE);
      const success = await writeBatchWithRetry(batch);

      if (!success) {
        // Re-queue failed batch for the next flush cycle
        failedLogs.push(...batch);
      } else {
        console.log(`Flushed ${batch.length} logs to the DB.`);
      }
    }

    // Prepend failed logs back to the front of the queue so they are retried first
    if (failedLogs.length > 0) {
      spend_logs = [...failedLogs, ...spend_logs];
      console.warn(
        `[flush] ${failedLogs.length} logs failed to write and were re-queued. Queue size: ${spend_logs.length}`
      );
    }

    const successCount = logsToProcess.length - failedLogs.length;
    console.log(
      `${successCount} logs processed successfully. Remaining in queue: ${spend_logs.length}`
    );
  } catch (error) {
    // Catch-all for unexpected errors so the flush loop never dies
    const errMsg = error instanceof Error ? error.message : String(error);
    console.error(`[flush] Unexpected error during flush cycle: ${errMsg}`);
    isDbHealthy = false;
  } finally {
    isFlushInProgress = false;
  }
};

// Setup interval for attempting to flush the logs
const flushTimer = setInterval(flushLogsToDb, FLUSH_INTERVAL);

app.get('/', (c) => {
  return c.text('Hello Hono!')
})

/**
 * Health check endpoint for Kubernetes liveness/readiness probes.
 * Returns 200 if the service is operational, 503 if the DB is unreachable.
 */
app.get('/health', async (c) => {
  try {
    await prisma.$queryRaw`SELECT 1`
    isDbHealthy = true;
    return c.json({
      status: 'healthy',
      queue_size: spend_logs.length,
      db_connected: true,
    })
  } catch (error) {
    isDbHealthy = false;
    const errMsg = error instanceof Error ? error.message : String(error);
    return c.json(
      {
        status: 'unhealthy',
        queue_size: spend_logs.length,
        db_connected: false,
        error: errMsg,
      },
      503
    )
  }
})

// Route to receive log messages
app.post('/spend/update', async (c) => {
  const incomingLogs = await c.req.json<LiteLLM_SpendLogs[]>();

  // Enforce queue size limit to prevent unbounded memory growth under sustained DB failures
  if (spend_logs.length + incomingLogs.length > MAX_QUEUE_SIZE) {
    const overflow = (spend_logs.length + incomingLogs.length) - MAX_QUEUE_SIZE;
    // Drop the oldest logs to make room
    spend_logs = spend_logs.slice(overflow);
    console.warn(
      `[queue] Queue size limit (${MAX_QUEUE_SIZE}) reached. Dropped ${overflow} oldest logs to make room.`
    );
  }

  spend_logs.push(...incomingLogs);

  console.log(`Received and stored ${incomingLogs.length} logs. Total logs in memory: ${spend_logs.length}`);

  return c.json({ message: `Successfully stored ${incomingLogs.length} logs` });
});

const port = 3000
console.log(`Server is running on port ${port}`)

const server = serve({
  fetch: app.fetch,
  port
})

/**
 * Graceful shutdown: flush remaining logs and release DB connections.
 * Prevents connection leaks when pods are restarted under load.
 */
async function gracefulShutdown(signal: string) {
  console.log(`\n[shutdown] Received ${signal}. Starting graceful shutdown...`);

  // Stop accepting new flush cycles
  clearInterval(flushTimer);

  // Attempt one final flush of remaining logs
  if (spend_logs.length > 0) {
    console.log(`[shutdown] Flushing ${spend_logs.length} remaining logs...`);
    try {
      await flushLogsToDb();
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : String(error);
      console.error(`[shutdown] Final flush failed: ${errMsg}`);
    }
  }

  // Disconnect Prisma to release all pooled connections
  try {
    await prisma.$disconnect();
    console.log('[shutdown] Prisma disconnected successfully.');
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : String(error);
    console.error(`[shutdown] Prisma disconnect error: ${errMsg}`);
  }

  process.exit(0);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));
