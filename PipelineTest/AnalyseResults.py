"""Backward-compatible re-export — module moved to PipelineTest.features.io"""
from PipelineTest.features.io import *  # noqa: F401,F403
from PipelineTest.features.io import (
    mergeOutputsList,
    GetInfoFromIndex,
    PlotInfo,
    PlotInfoRemasking,
    PlotPipeline,
    PlotScatterVarMaskedEntropyAcrossTokensVsAlphaEntropyAvg,
    PlotMeanVarMaskedAcrossTokens,
    PlotMeanMaskedEntropyAcrossTokens,
    PlotDistributionTau,
    plotEntropyForSteps,
    plotMeanEntropyAcrossTimeIncreaseOrder,
    plotMeanEntropyAndVarEntropyWithFittedCurve,
    plotEntropyAndLogProbsAcrossTime,
    getEntropicCost,
)
