"""Project-wide constants."""

TRADING_DAYS = 252      # Trading days in one year
N_SIMULATIONS = 1000    # Number of independent price paths
HIST_PERIOD = "1y"      # Historical window used for parameter estimation
EXCHANGE_SUFFIXES = (".NS", ".BO")  # NSE first, then BSE as fallback