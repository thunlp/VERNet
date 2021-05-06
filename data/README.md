# Data of VERNet Experiments
* The ``train`` and ``dev`` files are used as training set and development set during training VERNet. ``fce``, ``conll14.0`` and ``conll14.1`` are files for evaluation. ``conll14.0`` and ``conll14.1``are two annotaions in CoNLL-2014 dataset. Here is the format of these files:
```
{
	"src": Input sentence, 
	"src_lab": Grammatical error detection labels of input sentence,
	"hyp": GEC hypotheses from basic GEC model,
	"hyp_lab": GEC quality annotation labels of GEC hypotheses
}
```
* The ``conll14.m2`` and ``test.m2`` files contain the golden references of the CoNLL-2014 dataset. ``test.m2`` is the original file and ``conll14.m2`` is generated with ``ERRANT`` toolkit. ``fce.m2`` contains the golden references of FCE dataset and is also generated with the ``ERRANT`` toolkit.

* The ``conll14.src`` and ``fce.src`` files contain the source sentences from the CoNLL-2014 dataset and FCE dataset.
