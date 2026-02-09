"""
Test master key authentication when database is unavailable.

This test verifies the fix for issue #20427:
- Intermittent 401 Unauthorized for Master Key under load
- "ALL CONNECTION ATTEMPT FAILED" errors

The fix ensures that master key authentication works even when prisma_client is None,
preventing 401 errors during database connection issues.
"""

import asyncio
import secrets
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request

from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


@pytest.fixture
def mock_request():
    """Create a mock FastAPI Request object"""
    request = MagicMock(spec=Request)
    request.url = MagicMock()
    request.url.path = "/chat/completions"
    request.headers = {}
    request.query_params = {}

    async def mock_body():
        return {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}

    request.body = mock_body
    return request


@pytest.mark.asyncio
async def test_master_key_auth_with_db_unavailable(mock_request):
    """
    Test that master key authentication succeeds even when prisma_client is None.

    This simulates the scenario from issue #20427 where database connection
    failures cause intermittent 401 errors for valid master keys.
    """
    master_key = "sk-test-master-key-12345"
    test_api_key = f"Bearer {master_key}"

    # Mock the proxy_server imports
    with patch("litellm.proxy.auth.user_api_key_auth.prisma_client", None), \
         patch("litellm.proxy.auth.user_api_key_auth.master_key", master_key), \
         patch("litellm.proxy.auth.user_api_key_auth.user_api_key_cache") as mock_cache, \
         patch("litellm.proxy.auth.user_api_key_auth.proxy_logging_obj") as mock_logging, \
         patch("litellm.proxy.auth.user_api_key_auth.litellm_proxy_admin_name", "admin"), \
         patch("litellm.proxy.auth.user_api_key_auth.user_custom_auth", None), \
         patch("litellm.proxy.auth.user_api_key_auth.general_settings", {}), \
         patch("litellm.proxy.auth.user_api_key_auth.jwt_handler") as mock_jwt:

        # Configure mocks
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_logging.slack_alerting_instance = None
        mock_jwt.is_jwt = MagicMock(return_value=False)

        # Call the authentication function
        result = await user_api_key_auth(
            request=mock_request,
            api_key=test_api_key,
            azure_api_key_header=None,
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={"model": "gpt-4"},
            custom_litellm_key_header=None,
        )

        # Verify the result
        assert isinstance(result, UserAPIKeyAuth)
        assert result.api_key == master_key
        assert result.user_role.value == "proxy_admin"


@pytest.mark.asyncio
async def test_non_master_key_with_db_unavailable_raises_error(mock_request):
    """
    Test that non-master-key authentication fails appropriately when DB is unavailable.

    This ensures that the fix doesn't inadvertently allow unauthorized access.
    """
    master_key = "sk-test-master-key-12345"
    wrong_api_key = "Bearer sk-wrong-key-67890"

    # Mock the proxy_server imports
    with patch("litellm.proxy.auth.user_api_key_auth.prisma_client", None), \
         patch("litellm.proxy.auth.user_api_key_auth.master_key", master_key), \
         patch("litellm.proxy.auth.user_api_key_auth.user_api_key_cache") as mock_cache, \
         patch("litellm.proxy.auth.user_api_key_auth.proxy_logging_obj") as mock_logging, \
         patch("litellm.proxy.auth.user_api_key_auth.litellm_proxy_admin_name", "admin"), \
         patch("litellm.proxy.auth.user_api_key_auth.user_custom_auth", None), \
         patch("litellm.proxy.auth.user_api_key_auth.general_settings", {}), \
         patch("litellm.proxy.auth.user_api_key_auth.jwt_handler") as mock_jwt:

        # Configure mocks
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_logging.slack_alerting_instance = None
        mock_jwt.is_jwt = MagicMock(return_value=False)

        # Call the authentication function and expect it to raise
        with pytest.raises(ProxyException) as exc_info:
            await user_api_key_auth(
                request=mock_request,
                api_key=wrong_api_key,
                azure_api_key_header=None,
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data={"model": "gpt-4"},
                custom_litellm_key_header=None,
            )

        # Verify the exception
        assert exc_info.value.type == ProxyErrorTypes.no_db_connection
        assert "No connected db" in exc_info.value.message


@pytest.mark.asyncio
async def test_master_key_with_no_master_key_set(mock_request):
    """
    Test behavior when master_key is None (no master key configured).

    This verifies that requests are allowed when no master key is configured,
    which is the expected behavior for open proxies.
    """
    test_api_key = "Bearer sk-any-key-12345"

    # Mock the proxy_server imports with master_key = None
    with patch("litellm.proxy.auth.user_api_key_auth.prisma_client", None), \
         patch("litellm.proxy.auth.user_api_key_auth.master_key", None), \
         patch("litellm.proxy.auth.user_api_key_auth.user_api_key_cache") as mock_cache, \
         patch("litellm.proxy.auth.user_api_key_auth.proxy_logging_obj") as mock_logging, \
         patch("litellm.proxy.auth.user_api_key_auth.litellm_proxy_admin_name", "admin"), \
         patch("litellm.proxy.auth.user_api_key_auth.user_custom_auth", None), \
         patch("litellm.proxy.auth.user_api_key_auth.general_settings", {}), \
         patch("litellm.proxy.auth.user_api_key_auth.jwt_handler") as mock_jwt:

        # Configure mocks
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_logging.slack_alerting_instance = None
        mock_jwt.is_jwt = MagicMock(return_value=False)

        # Call the authentication function
        result = await user_api_key_auth(
            request=mock_request,
            api_key=test_api_key,
            azure_api_key_header=None,
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={"model": "gpt-4"},
            custom_litellm_key_header=None,
        )

        # Verify the result - should allow request as proxy admin
        assert isinstance(result, UserAPIKeyAuth)
        assert result.user_role.value == "proxy_admin"


@pytest.mark.asyncio
async def test_master_key_constant_time_comparison():
    """
    Test that master key comparison uses secrets.compare_digest for security.

    This prevents timing attacks on the master key.
    """
    master_key = "sk-test-master-key-12345"
    test_api_key = f"Bearer {master_key}"

    mock_request = MagicMock(spec=Request)
    mock_request.url = MagicMock()
    mock_request.url.path = "/chat/completions"
    mock_request.headers = {}
    mock_request.query_params = {}

    async def mock_body():
        return {"model": "gpt-4"}

    mock_request.body = mock_body

    with patch("litellm.proxy.auth.user_api_key_auth.prisma_client", None), \
         patch("litellm.proxy.auth.user_api_key_auth.master_key", master_key), \
         patch("litellm.proxy.auth.user_api_key_auth.user_api_key_cache") as mock_cache, \
         patch("litellm.proxy.auth.user_api_key_auth.proxy_logging_obj") as mock_logging, \
         patch("litellm.proxy.auth.user_api_key_auth.litellm_proxy_admin_name", "admin"), \
         patch("litellm.proxy.auth.user_api_key_auth.user_custom_auth", None), \
         patch("litellm.proxy.auth.user_api_key_auth.general_settings", {}), \
         patch("litellm.proxy.auth.user_api_key_auth.jwt_handler") as mock_jwt, \
         patch("litellm.proxy.auth.user_api_key_auth.secrets.compare_digest") as mock_compare:

        # Configure mocks
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_logging.slack_alerting_instance = None
        mock_jwt.is_jwt = MagicMock(return_value=False)
        mock_compare.return_value = True

        # Call the authentication function
        result = await user_api_key_auth(
            request=mock_request,
            api_key=test_api_key,
            azure_api_key_header=None,
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={"model": "gpt-4"},
            custom_litellm_key_header=None,
        )

        # Verify that secrets.compare_digest was called
        # It should be called at least once (in the early check and/or main check)
        assert mock_compare.call_count >= 1

        # Verify first call had the correct arguments
        first_call_args = mock_compare.call_args_list[0][0]
        assert master_key in first_call_args
