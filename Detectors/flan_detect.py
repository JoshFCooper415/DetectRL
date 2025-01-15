from typing import Union, Optional
import os
import numpy as np
import torch
import transformers
from transformers import T5ForConditionalGeneration, AutoTokenizer, PreTrainedTokenizer

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

torch.set_grad_enabled(False)

huggingface_config = {
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

BINOCULARS_ACCURACY_THRESHOLD = 4.4
BINOCULARS_FPR_THRESHOLD = 5

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

def assert_tokenizer_consistency(model_id_1, model_id_2):
    identical_tokenizers = (
            AutoTokenizer.from_pretrained(model_id_1).vocab
            == AutoTokenizer.from_pretrained(model_id_2).vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")

class Binoculars(object):
    def __init__(self,
                 observer_name_or_path: str = "google/flan-t5-small",
                 performer_name_or_path: str = "vineetsharma/dialogsum-flan-t5-small",  # Replace with actual path
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.change_mode(mode)
        self.observer_model = T5ForConditionalGeneration.from_pretrained(
            observer_name_or_path,
            device_map={"": DEVICE_1},
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            token=huggingface_config["TOKEN"]
        )
        self.performer_model = T5ForConditionalGeneration.from_pretrained(
            performer_name_or_path,
            device_map={"": DEVICE_2},
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            token=huggingface_config["TOKEN"]
        )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    @staticmethod
    def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        return logits / temperature

    @staticmethod
    def apply_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
        values, indices = torch.topk(probs, k=k)
        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(-1, indices, values)
        return filtered_probs

    @staticmethod
    def apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(-1, sorted_indices, sorted_probs * (~sorted_indices_to_remove))
        return filtered_probs

    @staticmethod
    def apply_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
        score = torch.gather(logits, -1, input_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(-1, input_ids, score)
        return logits

    @torch.inference_mode()
    def compute_telescope_one_forward(
        self,
        reference_text: str,
        performer_model: T5ForConditionalGeneration,
        observer_model: T5ForConditionalGeneration,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        temperature: float = .7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 2
        ):
        
        # Prepare input for T5 models
        input_ids = tokenizer(f"analyze text: {reference_text}", 
                            return_tensors="pt",
                            max_length=self.max_token_observed,
                            truncation=True).to(device).input_ids
        
        # Generate a target sequence for comparison
        target_ids = input_ids.clone()  # Use input as target for perplexity calculation
        
        # Get logits from both models
        performer_outputs = performer_model(
            input_ids=input_ids,
            labels=target_ids,
            output_hidden_states=True,
            return_dict=True
        )
        observer_outputs = observer_model(
            input_ids=input_ids,
            labels=target_ids,
            output_hidden_states=True,
            return_dict=True
        )

        # Get the logits from the last hidden state
        performer_logits = performer_outputs.logits[0]  # Shape: [seq_len, vocab_size]
        observer_logits = observer_outputs.logits[0]    # Shape: [seq_len, vocab_size]
        
        total_cross_entropy_cross_perplexity = 0
        total_cross_entropy_normal_perplexity = 0
        
        # Process only non-padded tokens
        seq_len = (input_ids[0] != tokenizer.pad_token_id).sum()
        
        for i in range(seq_len - 1):  # -1 because we're predicting next tokens
            performer_next_token_logits = performer_logits[i].unsqueeze(0)  # Shape: [1, vocab_size]
            observer_next_token_logits = observer_logits[i].unsqueeze(0)    # Shape: [1, vocab_size]

            if repetition_penalty != 1.0:
                performer_next_token_logits = self.apply_repetition_penalty(
                    performer_next_token_logits, 
                    input_ids[0, :i+1], 
                    repetition_penalty
                )
                observer_next_token_logits = self.apply_repetition_penalty(
                    observer_next_token_logits,
                    input_ids[0, :i+1],
                    repetition_penalty
                )

            if temperature != 1.0:
                performer_next_token_logits = self.apply_temperature(performer_next_token_logits, temperature)
                observer_next_token_logits = self.apply_temperature(observer_next_token_logits, temperature)

            performer_next_tokens_logits_softmax = torch.softmax(performer_next_token_logits, dim=-1)
            observer_next_token_logits_softmax = torch.softmax(observer_next_token_logits, dim=-1)

            if top_k is not None:
                performer_next_tokens_logits_softmax = self.apply_top_k(performer_next_tokens_logits_softmax, top_k)
                observer_next_token_logits_softmax = self.apply_top_k(observer_next_token_logits_softmax, top_k)

            if top_p is not None:
                performer_next_tokens_logits_softmax = self.apply_top_p(performer_next_tokens_logits_softmax, top_p)
                observer_next_token_logits_softmax = self.apply_top_p(observer_next_token_logits_softmax, top_p)

            if top_k is not None or top_p is not None:
                performer_next_tokens_logits_softmax = performer_next_tokens_logits_softmax / performer_next_tokens_logits_softmax.sum()
                observer_next_token_logits_softmax = observer_next_token_logits_softmax / observer_next_token_logits_softmax.sum()
            
            # Calculate cross-entropy and perplexity
            total_cross_entropy_cross_perplexity -= torch.matmul(
                performer_next_tokens_logits_softmax, 
                torch.log(observer_next_token_logits_softmax).T
            )
            total_cross_entropy_normal_perplexity -= torch.log(
                performer_next_tokens_logits_softmax[:, target_ids[0, i+1]]
            )
            
            del observer_next_token_logits_softmax
            del performer_next_tokens_logits_softmax
            torch.cuda.empty_cache()
            
        result = (total_cross_entropy_normal_perplexity / total_cross_entropy_cross_perplexity).to(torch.float32).cpu().numpy()
        return float(result)

    def compute_score(self, 
                     input_text: Union[list[str], str],
                     temperature: float = 1.0,
                     top_k: Optional[int] = None,
                     top_p: Optional[float] = None,
                     repetition_penalty: float = 1.0
                     ) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        scores = [self.compute_telescope_one_forward(
            text,
            self.performer_model,
            self.observer_model,
            self.tokenizer,
            self.performer_model.device,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        ) for text in batch]
        return scores[0] if isinstance(input_text, str) else scores

    def predict(self, 
                input_text: Union[list[str], str],
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                repetition_penalty: float = 1.0
                ) -> Union[list[str], str]:
        binoculars_scores = np.array(self.compute_score(
            input_text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        ))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred[0] if isinstance(input_text, str) else pred