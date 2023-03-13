# vibeOptimiser

Uses NLOPT (https://github.com/DanielBok/nlopt-python) to optimise parameters of Sky360Lib Vibe

This optimisation process could be triggered by changes in cloud cover / weather conditions e.g. take cloud cover estimate every few mins and if cloud cover deviates by X amount run an optimisation algorithm over N frames of video to determine optimal Vibe parameters for the conditions 
