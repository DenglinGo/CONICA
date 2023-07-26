from utils.chair import *

import clip
print(clip.load("RN50x4"))
#
_, imids = load_generated_captions("result.json")
evaluator = CHAIR(imids, "/home/smart-solution-server-001/Documents/dataset_fast/caption/mscoco/annotations")
evaluator.get_annotations()
print()

cap_dict = evaluator.compute_chair("result_noscd.json")
print_metrics(cap_dict)

cap_dict = evaluator.compute_chair("result_SCD.json")
print_metrics(cap_dict)

cap_dict = evaluator.compute_chair("result_xe.json")
print_metrics(cap_dict)

cap_dict = evaluator.compute_chair("result.json")
print_metrics(cap_dict)