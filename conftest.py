import os

"""
conftest.py — provide safe default environment variables for tests so
test collection doesn't fail when API keys are not set locally/CI.
Replace these defaults with real secrets only when running integration tests.
"""

# Only set defaults if not already provided in the environment
os.environ.setdefault("BINANCE_API_KEY", "test_binance_api_key")
os.environ.setdefault("BINANCE_API_SECRET", "test_binance_api_secret")
# If tests need testnet flags, you can add them too
os.environ.setdefault("BINANCE_TESTNET", "true")
