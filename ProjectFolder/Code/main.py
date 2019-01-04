
import argparse

#   TODO:   [all]       i. Init everything, namely: noise level


#   TODO:   [Andreas]   ii. CQT calculate [advTrain]


#   TODO:	[Sebastian] iii. Train base model (for a given number of epochs, with intermed. Result saved) kerasTrain ->parameter reduzieren]



#   TODO:   [Sebastian] iv.	Save params


#   TODO:   [all]        v.	For noiseEpochs = 1 … XXX
#   TODO:   [Malte,Andreas] 1.	While true
#                                a.	Generate noise candidate (only) with current noise level [advTrain]
#                                b.	Combine noise with clean data (noise and audio)
#                                c.	CQT
#                                d.	Evaluate performance of classifier based on noise candidate
#                                e.	If  (“AMT sufficiently failing (interval!)” -> define threshold)
#                                i.	Break -> save Noise audio file
#                                f.	Else if “too easy for AMT” (define deviation of score) -> increase noise level
#                                g.	Else if “too hard for AMT” -> decrease noise level
#   TODO:   [Tanos]         2.	Train with noisy samples (for a given number of epochs, with intermed. Result saved)
#   TODO:   [Tanos]         3.	Save intermediate results
#                           4.	Go to noise candidate


#   TODO:   [all]       vi.	Overall eval:
#                           1.	F1 score compared to noise level
#                           2.	Confusion matrix (heat maps, for e.g. 4 noise levels)
