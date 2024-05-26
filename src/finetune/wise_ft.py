import os

import numpy as np

import torch

from src.finetune.finetune import finetune
from src.finetune.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from src.finetune.eval import evaluate

def _merge(alpha, theta_0, theta_1):
    # interpolate between all weights in the checkpoints
    return {
        key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0.keys()
    }


def wise_ft(args, model, preprocess_train, preprocess_val, tokenizer):
    args.save = os.path.join(args.logs, args.name)
    
    if args.load is None:
        # Build and save zero-shot model
        image_encoder = ImageEncoder(args, model, preprocess_train, preprocess_val, keep_lang=True)
        zeroshot_weights = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )
        zeroshot_weights *= model.logit_scale.exp()
        classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights.T)
        delattr(image_encoder.model, 'transformer')
        classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
        zeroshot_checkpoint = os.path.join(args.save, 'zeroshot.pt')
        classifier.save(zeroshot_checkpoint)

        # Standard fine-tuning
        args.load = zeroshot_checkpoint
        args.save = os.path.join(args.save, 'finetuned')
        finetuned_checkpoint = finetune(args)
    else:
        # No need to compute things from stratch
        assert len(args.load) == 2
        zeroshot_checkpoint, finetuned_checkpoint = args.load

    # Load models
    zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    finetuned = ImageClassifier.load(finetuned_checkpoint)
    theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}
    del zeroshot

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    alphas = args.alpha
    for alpha in alphas:
        args.alpha = alpha

        theta = _merge(alpha, theta_0, theta_1)

        # update the model (in-place) acccording to the new weights
        finetuned.load_state_dict(theta)

        # save model
        finetuned.save(os.path.join(args.save, f'wise_ft_alpha={alpha:.3f}.pt'))

        # evaluate
        evaluate(finetuned, args)

