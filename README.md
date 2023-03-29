# bgs-optimizer

Uses NLOPT (https://github.com/DanielBok/nlopt-python) to optimize parameters of sky360lib background subtraction algorithms

This optimization process could be triggered by changes in cloud cover / weather conditions e.g. take cloud cover estimate every few mins and if cloud cover deviates by X amount run an optimization algorithm over N frames of video to try and determine optimal parameters for the conditions.

Fitness is based on Matthews Correlation Coefficient. Could be switched to f-score in objective function. Or we could come up with our own which has multiple objectives that are weighted in terms of importance e.g. precision weighted more heavily over recall

Examples for Vibe and WMV included
