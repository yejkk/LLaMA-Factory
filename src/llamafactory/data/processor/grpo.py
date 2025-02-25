from .processor_utils import DatasetProcessor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict


class GRPODatasetProcessor(DatasetProcessor):

    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            prompt = []
            system=examples["_system"][i]
            if system is not None:
                prompt.append({"role": "system", "content": system})

            prompt.extend(examples["_prompt"][i])
            model_inputs['prompt'].append(prompt)
            model_inputs['_response'].append(examples["_response"][i])
        return model_inputs

    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        print("prompt:\n{}".format(example["prompt"]))