import json
import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

def load_model():
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = 'left'
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        max_new_tokens=10,
        do_sample=False,
        batch_size=32  # Process 32 questions at once
    )
    
    return generator

def create_prompt(question, answer):
    return f"""You are filtering Q&A pairs for CT scan training data.
ONLY keep questions that can be answered by looking at a single head CT scan image.

DISCARD questions that:
- Reference the report or "noted" findings (e.g., "what is noted", "what changes are noted")
- Require comparison to prior imaging
- Require clinical context or history
- Synthesize findings across multiple body regions
- Are phrased as testing knowledge of the report text

KEEP questions that:
- Ask what can be seen/observed/visualized on the scan
- Ask about presence/absence of specific findings
- Ask about measurements or characteristics visible on images

When in doubt, DISCARD.

Question: {question}
Answer: {answer}

Can this be answered by looking at the CT scan alone? Respond with only KEEP or DISCARD:"""

def filter_questions_batch(generator, prompts):
    """Process a batch of prompts in parallel"""
    responses = generator(prompts, max_new_tokens=10, do_sample=False)
    
    decisions = []
    for i, response in enumerate(responses):
        # Extract just the model's response (after the prompt)
        decision = response[0]['generated_text'][len(prompts[i]):].strip().upper()
        decisions.append("KEEP" in decision)
    
    return decisions

def process_reports(input_file, output_file, batch_size=32, checkpoint_file=None):
    # Load data
    print(f"Reading from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Collect all questions with metadata
    all_questions = []
    for report_idx, report in enumerate(data):
        # Skip if qa is None or missing
        if report.get('qa') is None:
            continue
        if 'items' not in report['qa'] or report['qa']['items'] is None:
            continue
        
        for item_idx, item in enumerate(report['qa']['items']):
            all_questions.append({
                'report_idx': report_idx,
                'item_idx': item_idx,
                'question': item['q'],
                'answer': item['a'],
                'item': item
            })
    
    total_questions = len(all_questions)
    print(f"Total questions to process: {total_questions}")
    
    # Try to load checkpoint
    start_idx = 0
    kept_flags = []
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Found checkpoint file: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint['processed']
            kept_flags = checkpoint['kept_flags']
        print(f"Resuming from checkpoint: {start_idx}/{total_questions} questions processed")
    
    # Load model (only if we have questions to process)
    if start_idx < total_questions:
        generator = load_model()
    else:
        print("All questions already processed!")
        generator = None
    
    # Process in batches
    if generator is not None:
        print("Processing questions in batches...")
        
        for i in tqdm(range(start_idx, len(all_questions), batch_size)):
            batch = all_questions[i:i+batch_size]
            prompts = [create_prompt(q['question'], q['answer']) for q in batch]
            
            batch_decisions = filter_questions_batch(generator, prompts)
            kept_flags.extend(batch_decisions)
            
            # Save checkpoint every 10 batches
            if checkpoint_file and (i // batch_size) % 10 == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'processed': i + len(batch),
                        'kept_flags': kept_flags
                    }, f)
            
            # Optional: print some feedback
            for j, (q, keep) in enumerate(zip(batch, batch_decisions)):
                if (i + j) % 100 == 0:  # Print every 100th question
                    status = "✓ KEPT" if keep else "✗ DISCARDED"
                    print(f"{status}: {q['question'][:60]}...")
    
    # Save final checkpoint
    if checkpoint_file:
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'processed': total_questions,
                'kept_flags': kept_flags
            }, f)
    
    # Build filtered dataset
    print("\nBuilding filtered dataset...")
    filtered_data = []
    kept_questions = 0
    
    # Group by report
    report_items = {}
    for q, keep in zip(all_questions, kept_flags):
        if keep:
            kept_questions += 1
            report_idx = q['report_idx']
            if report_idx not in report_items:
                report_items[report_idx] = []
            report_items[report_idx].append(q['item'])
    
    # Build filtered reports - preserve all fields from original
    for report_idx, items in report_items.items():
        report_copy = data[report_idx].copy()
        report_copy['qa'] = {'items': items}
        filtered_data.append(report_copy)
    
    # Save filtered data
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Total reports in input: {len(data)}")
    print(f"Reports with valid QA: {len([r for r in data if r.get('qa') is not None and r['qa'].get('items') is not None])}")
    print(f"Total questions: {total_questions}")
    print(f"Kept: {kept_questions}")
    print(f"Discarded: {total_questions - kept_questions}")
    print(f"Kept percentage: {100*kept_questions/total_questions:.1f}%")
    print(f"Reports with questions after filtering: {len(filtered_data)}")
    print(f"Saved to: {output_file}")
    
    # Clean up checkpoint file if successful
    if checkpoint_file and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Removed checkpoint file: {checkpoint_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_qa_parallel.py <input_file> [output_file] [batch_size] [checkpoint_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "radiology_qa_final.json"
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    checkpoint_file = sys.argv[4] if len(sys.argv) > 4 else "filter_checkpoint.json"
    
    process_reports(input_file, output_file, batch_size, checkpoint_file)