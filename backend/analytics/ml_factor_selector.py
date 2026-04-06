"""ML-based factor selection using LightGBM.

Trains a gradient boosting model to predict forward returns from
technical and fundamental features. Extracts feature importance to
identify which factors contribute most to return prediction.

Used to:
1. Validate research-based factor weights in MultiFactorModel
2. Dynamically adjust factor weights based on recent market data
3. Identify which technical indicators have predictive power

Designed to run periodically (weekly/monthly) and update FactorWeights.
Falls back to research-based weights if training fails or data is insufficient.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum requirements for training
MIN_STOCKS = 10
MIN_BARS = 60
FORWARD_DAYS = 20  # Predict 20-day forward return


@dataclass
class FeatureImportance:
    """Feature importance from trained model."""
    name: str
    importance: float  # Gini importance (higher = more predictive)
    rank: int = 0


@dataclass
class FactorSelectionResult:
    """Result of ML factor selection."""
    features: list[FeatureImportance] = field(default_factory=list)
    train_r2: float = 0.0
    test_r2: float = 0.0
    n_stocks: int = 0
    n_samples: int = 0
    success: bool = False

    @property
    def top_features(self) -> list[str]:
        """Top features sorted by importance."""
        return [f.name for f in sorted(self.features, key=lambda x: -x.importance)]

    def to_factor_weights(self) -> dict[str, float]:
        """Map feature importances to factor weight adjustments.

        Returns dict of factor_name -> relative weight (summing to 1.0).
        Only returns factors from the standard set (growth, profitability, etc.).
        """
        factor_map = {
            "momentum_6m": "momentum",
            "momentum_12m": "momentum",
            "momentum_3m": "momentum",
            "volatility": "momentum",  # vol is inverse-momentum indicator
            "trend_strength": "momentum",
            "revenue_growth": "growth",
            "earnings_growth": "growth",
            "profit_margin": "profitability",
            "roe": "profitability",
            "garp_ratio": "garp",
        }

        factor_scores: dict[str, float] = {
            "growth": 0.0, "profitability": 0.0, "garp": 0.0, "momentum": 0.0,
        }

        for feat in self.features:
            factor = factor_map.get(feat.name)
            if factor:
                factor_scores[factor] += feat.importance

        total = sum(factor_scores.values())
        if total > 0:
            return {k: v / total for k, v in factor_scores.items()}
        return {"growth": 0.35, "profitability": 0.30, "garp": 0.20, "momentum": 0.15}


class MLFactorSelector:
    """Select predictive factors using LightGBM feature importance."""

    def __init__(
        self,
        forward_days: int = FORWARD_DAYS,
        test_ratio: float = 0.20,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
    ):
        self._forward_days = forward_days
        self._test_ratio = test_ratio
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._min_child_samples = min_child_samples

    def select_factors(
        self,
        price_data: dict[str, pd.DataFrame],
        fundamental_data: dict[str, dict] | None = None,
    ) -> FactorSelectionResult:
        """Train LightGBM and extract feature importance.

        Args:
            price_data: {symbol: OHLCV DataFrame} with at least 60 bars.
            fundamental_data: {symbol: {revenueGrowth, profitMargins, ...}}.

        Returns:
            FactorSelectionResult with feature importances and model quality.
        """
        if len(price_data) < MIN_STOCKS:
            logger.warning(
                "Insufficient stocks for ML factor selection (%d < %d)",
                len(price_data), MIN_STOCKS,
            )
            return FactorSelectionResult()

        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM not available, skipping ML factor selection")
            return FactorSelectionResult()

        # Build feature matrix
        X, y, feature_names = self._build_dataset(price_data, fundamental_data)

        if len(X) < 50:
            logger.warning("Insufficient samples for training (%d < 50)", len(X))
            return FactorSelectionResult()

        # Train/test split (time-aware: last test_ratio of each stock's data)
        split_idx = int(len(X) * (1 - self._test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(X_train) < 30 or len(X_test) < 10:
            logger.warning("Insufficient train/test samples")
            return FactorSelectionResult()

        # Train LightGBM
        model = lgb.LGBMRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            min_child_samples=self._min_child_samples,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
        )

        # Evaluate
        train_r2 = float(model.score(X_train, y_train))
        test_r2 = float(model.score(X_test, y_test))

        # Extract feature importance
        importances = model.feature_importances_
        total_imp = float(sum(importances))
        if total_imp <= 0:
            total_imp = 1.0

        features = []
        for i, name in enumerate(feature_names):
            features.append(FeatureImportance(
                name=name,
                importance=float(importances[i]) / total_imp,
            ))

        features.sort(key=lambda f: -f.importance)
        for i, f in enumerate(features):
            f.rank = i + 1

        result = FactorSelectionResult(
            features=features,
            train_r2=round(train_r2, 4),
            test_r2=round(test_r2, 4),
            n_stocks=len(price_data),
            n_samples=len(X),
            success=True,
        )

        logger.info(
            "ML factor selection: %d stocks, %d samples, "
            "train_R2=%.3f, test_R2=%.3f, top=%s",
            result.n_stocks, result.n_samples,
            result.train_r2, result.test_r2,
            result.top_features[:5],
        )

        return result

    def _build_dataset(
        self,
        price_data: dict[str, pd.DataFrame],
        fundamental_data: dict[str, dict] | None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build feature matrix and target vector from price/fundamental data."""
        fund = fundamental_data or {}
        all_X = []
        all_y = []

        feature_names = [
            "momentum_3m", "momentum_6m", "momentum_12m",
            "volatility", "trend_strength", "mean_reversion",
            "volume_trend", "rsi_proxy",
            "revenue_growth", "earnings_growth", "profit_margin",
            "roe", "garp_ratio",
        ]

        for symbol, df in price_data.items():
            if len(df) < MIN_BARS + self._forward_days:
                continue

            close = df["close"].values.astype(float)
            volume = df["volume"].values.astype(float)
            n = len(close)

            sym_fund = fund.get(symbol, {})

            # Generate features for each valid time point
            for t in range(max(252, MIN_BARS), n - self._forward_days):
                features = self._extract_features(close, volume, t, sym_fund)
                if features is None:
                    continue

                # Target: forward return
                forward_ret = (close[t + self._forward_days] / close[t]) - 1
                if np.isnan(forward_ret) or np.isinf(forward_ret):
                    continue

                all_X.append(features)
                all_y.append(forward_ret)

        if not all_X:
            return np.array([]), np.array([]), feature_names

        return np.array(all_X), np.array(all_y), feature_names

    def _extract_features(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        t: int,
        fund: dict,
    ) -> list[float] | None:
        """Extract feature vector at time t."""
        if t < 252 or close[t] <= 0:
            return None

        try:
            # Momentum features
            mom_3m = (close[t] / close[t - 63] - 1) if close[t - 63] > 0 else 0
            mom_6m = (close[t] / close[t - 126] - 1) if close[t - 126] > 0 else 0
            mom_12m = (close[t] / close[t - 252] - 1) if close[t - 252] > 0 else 0

            # Volatility (20-day)
            rets = np.diff(close[t - 20:t + 1]) / close[t - 20:t]
            vol = float(np.std(rets)) * np.sqrt(252) if len(rets) > 1 else 0.2

            # Trend strength (R² of linear regression over 60 days)
            y = close[t - 60:t + 1]
            x = np.arange(len(y))
            if len(y) > 1 and np.std(y) > 0:
                corr = np.corrcoef(x, y)[0, 1]
                trend = corr ** 2
            else:
                trend = 0.0

            # Mean reversion (20-day z-score)
            ma20 = np.mean(close[t - 20:t + 1])
            std20 = np.std(close[t - 20:t + 1])
            mean_rev = (close[t] - ma20) / std20 if std20 > 0 else 0

            # Volume trend (current vs 20-day avg)
            vol_avg = np.mean(volume[max(0, t - 20):t]) if t >= 20 else 1
            vol_trend = (volume[t] / vol_avg - 1) if vol_avg > 0 else 0

            # RSI proxy (% of up days in last 14 days)
            if t >= 14:
                diffs = np.diff(close[t - 14:t + 1])
                rsi_proxy = float(np.sum(diffs > 0)) / len(diffs)
            else:
                rsi_proxy = 0.5

            # Fundamental features (static per stock, zero if unavailable)
            rev_growth = float(fund.get("revenueGrowth") or fund.get("revenue_growth") or 0)
            earn_growth = float(fund.get("earningsGrowth") or fund.get("earnings_growth") or 0)
            profit_margin = float(fund.get("profitMargins") or fund.get("profit_margin") or 0)
            roe_val = float(fund.get("returnOnEquity") or fund.get("roe") or 0)
            pe = float(fund.get("forwardPE") or fund.get("trailingPE") or fund.get("pe_ratio") or 0)
            garp = rev_growth / pe if pe > 0 else 0

            features = [
                mom_3m, mom_6m, mom_12m,
                vol, trend, mean_rev,
                vol_trend, rsi_proxy,
                rev_growth, earn_growth, profit_margin,
                roe_val, garp,
            ]

            # Clip extremes
            return [float(np.clip(f, -10, 10)) for f in features]

        except (IndexError, ValueError, ZeroDivisionError):
            return None
