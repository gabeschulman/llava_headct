# Generate Q&A pairs from radiology impressions using GPU acceleration
# 
# Parameters:
#   -n: number of impressions to process (default: all records)
#   -s: starting index, skip first N records (default: 0)
#   -c: checkpoint interval (save progress every N records)
#   -b: batch size for GPU processing (default: 2)
#
# Usage examples:
#   python generate_qa3.py -n 100 -c 10 -b 2
#   python generate_qa3.py -c 1000 -b 4

import os
import polars as pl
import re, json
import argparse
from string import Template
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import gc

ct_radiology_notes = "/gpfs/data/razavianlab/capstone/2025_stroke/ct_accession_report_2506_deid.csv"

radiology_df = pl.read_csv(
    ct_radiology_notes,
    infer_schema_length=10000,
)

def strip_imp(s): return re.sub(r'^\s*IMPRESSION:\s*', '', str(s), flags=re.I).strip()

def combine_narrative_impression(narrative, impression):
    """Combine narrative and impression into a single text"""
    narrative_text = strip_imp(str(narrative)) if narrative else ""
    impression_text = strip_imp(str(impression)) if impression else ""
    
    # Combine with clear separation
    combined = ""
    if narrative_text:
        combined += f"NARRATIVE:\n{narrative_text}\n\n"
    if impression_text:
        combined += f"IMPRESSION:\n{impression_text}"
    
    return combined.strip()

def chat_prompt(msg, sys="You are a concise radiology assistant. Be clinical and factual."):
    return f"<|system|>\n{sys}\n<|user|>\n{msg}\n<|assistant|>\n"

FEW_SHOT = r"""
You will read a radiology HEAD CT report and create exactly 3 question-answer pairs.

CRITICAL RULES - READ CAREFULLY:
1. Questions MUST ask about visible imaging findings on the HEAD CT scan only
2. Answers MUST be direct, factual statements from the report
3. DO NOT ask about the patient history unless it is clear from the most recent HEAD CT scan
4. Use clinical terminology directly from the report
5. Keep answers to 1-2 sentences maximum

VALID question types:
✓ "Is there evidence of [finding] visible on the CT?"
✓ "Does the scan show [specific finding]?"
✓ "How would you describe the [anatomical structure] based on the CT findings?"
✓ "What does this suggest?"
✓ "What should be done next?"

INVALID question types:
✗ "What does the report say about the condition?" (references the report)
✗ "Has the patient improved?" (referring to previous findings)
✗ "What abnormality is described?" (references the report)
✗ "Is there any significant change observed between the current and previous brain CTs?" (references previous findings)
✗ "What changes are noted in [anatomical structure]?" (references previous findings)

Return ONLY valid JSON with this EXACT structure:
{"items":[{"q":"...","a":"..."},{"q":"...","a":"..."},{"q":"...","a":"..."}]}

Example REPORT:
NARRATIVE: CT head without contrast performed.
IMPRESSION: No acute intracranial hemorrhage. Old left basal ganglia lacunar infarct. Mild chronic microvascular ischemic changes.

CORRECT JSON:
{"items":[{"q":"Is there any acute bleeding visible on the CT?","a":"No acute intracranial hemorrhage is seen."}]}

WRONG - DO NOT DO THIS:
{"items":[{"q":"What does the report describe?","a":"..."}]}
"""

PROMPT_TMPL = Template(FEW_SHOT + """

Now generate for this REPORT:
$report

Return ONLY the JSON. No commentary.
""")

def extract_json(text):
    """Extract and parse JSON from model output"""
    # Try to find JSON in the text
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        return None
    
    try:
        return json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None

def validate_qa(qa_data):
    """Validate Q&A structure and clean it up"""
    if not qa_data or not isinstance(qa_data, dict):
        return None
    
    if "items" not in qa_data or not isinstance(qa_data["items"], list):
        return None
    
    # Clean up items - keep only q and a fields
    cleaned_items = []
    for item in qa_data["items"]:
        if not isinstance(item, dict):
            continue
        
        q = str(item.get("q", "")).strip()
        a = str(item.get("a", "")).strip()
        
        # Skip empty or very short Q&A
        if len(q) < 10 or len(a) < 10:
            continue
        
        cleaned_items.append({"q": q, "a": a})
    
    # Accept if we have at least 1 valid Q&A pair
    if len(cleaned_items) >= 1:
        return {"items": cleaned_items}
    
    return None

def gen_qa_batch(reports, generator):
    """Generate Q&A for a batch of reports using GPU"""
    prompts = [chat_prompt(PROMPT_TMPL.substitute(report=report)) for report in reports]
    
    # Generate in batch on GPU
    outputs = generator(prompts, max_new_tokens=512, temperature=0.15, top_p=0.9,
                       repetition_penalty=1.25, do_sample=True, batch_size=len(prompts))
    
    results = []
    for output in outputs:
        text = output[0]["generated_text"].split("<|assistant|>")[-1].strip()
        
        # Extract and validate JSON
        qa_data = extract_json(text)
        validated = validate_qa(qa_data)
        
        if validated:
            results.append({
                "qa": validated,
                "raw_output": text,
                "status": "success"
            })
        else:
            results.append({
                "qa": None,
                "raw_output": text,
                "status": "failed_validation"
            })
    
    return results

def save_checkpoint(results, checkpoint_file):
    """Save results to a checkpoint file"""
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Checkpoint saved: {len(results)} records written to {checkpoint_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate Q&A pairs from radiology impressions')
    parser.add_argument('--num', '-n', type=int, default=None,
                        help='Number of impressions to process (default: all records)')
    parser.add_argument('--start', '-s', type=int, default=0,
                        help='Starting index (default: 0)')
    parser.add_argument('--checkpoint-interval', '-c', type=int, default=100,
                        help='Save checkpoint every N records (default: 100)')
    parser.add_argument('--output', '-o', type=str, default='radiology_qa_results.json',
                        help='Output JSON file (default: radiology_qa_results.json)')
    parser.add_argument('--batch-size', '-b', type=int, default=2,
                        help='Batch size for GPU processing (default: 2)')
    
    args = parser.parse_args()
    
    start_index = args.start
    num_to_process = args.num if args.num is not None else len(radiology_df) - start_index
    checkpoint_interval = args.checkpoint_interval
    output_file = args.output
    batch_size = args.batch_size
    checkpoint_file = output_file.replace('.json', '_checkpoint.json')
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load model once on GPU
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    print(f"Starting from index {start_index}")
    print(f"Processing {num_to_process} reports...")
    print(f"Batch size: {batch_size}")
    print(f"Checkpoint will be saved every {checkpoint_interval} records to {checkpoint_file}")
    print(f"Final output will be saved to {output_file}")
    print("-" * 80)
    
    results = []
    success_count = 0
    failed_count = 0
    
    # Process in batches
    for batch_start in range(start_index, min(start_index + num_to_process, len(radiology_df)), batch_size):
        batch_end = min(batch_start + batch_size, start_index + num_to_process, len(radiology_df))
        
        # Prepare batch
        batch_data = []
        for i in range(batch_start, batch_end):
            row = radiology_df[i]
            accession_num = row["accession_num"][0]
            narrative = row["narrative_deid"][0]
            impression = row["impression_deid"][0]
            combined_report = combine_narrative_impression(narrative, impression)
            
            batch_data.append({
                "index": i,
                "accession_num": accession_num,
                "narrative": narrative,
                "impression": impression,
                "combined_report": combined_report
            })
        
        reports = [item["combined_report"] for item in batch_data]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}: records {batch_start + 1} to {batch_end}")
        
        # Generate Q&A for batch
        try:
            batch_results = gen_qa_batch(reports, generator)
            
            # Store results
            for item, result in zip(batch_data, batch_results):
                record = {
                    "accession_num": item["accession_num"],
                    "narrative": item["narrative"],
                    "impression": item["impression"],
                    "qa": result["qa"],
                    # "raw_output": result["raw_output"],
                    # "status": result["status"]
                }
                results.append(record)
                
                if result["status"] == "success":
                    success_count += 1
                    print(f"  ✓ [{item['index']+1}]: {item['accession_num']} - {len(result['qa']['items'])} Q&A pairs")
                else:
                    failed_count += 1
                    print(f"  ✗ [{item['index']+1}]: {item['accession_num']} - validation failed")
        
        except Exception as e:
            print(f"  Batch error: {str(e)}")
            # Store error
            for item in batch_data:
                results.append({
                    "accession_num": item["accession_num"],
                    "narrative": item["narrative"],
                    "impression": item["impression"],
                    "qa": None,
                    # "raw_output": f"ERROR: {str(e)}",
                    # "status": "error"
                })
                failed_count += 1
        
        # Save checkpoint at intervals
        if len(results) % checkpoint_interval <= batch_size and len(results) > 0:
            save_checkpoint(results, checkpoint_file)
            print(f"Progress: {batch_end - start_index}/{num_to_process} completed")
            print(f"Success: {success_count}, Failed: {failed_count}")
    
    # Save final output
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'-'*80}")
    print(f"Processing complete!")
    print(f"Total: {len(results)} records")
    print(f"Success: {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"Failed: {failed_count} ({failed_count/len(results)*100:.1f}%)")
    print(f"Saved to {output_file}")
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        print(f"Removing checkpoint file {checkpoint_file}")
        os.remove(checkpoint_file)
    
    # Show sample success and failure
    if success_count > 0:
        success_example = next((r for r in results if r["status"] == "success"), None)
        if success_example:
            print(f"\nExample SUCCESS:")
            print(f"Accession: {success_example['accession_num']}")
            print(json.dumps(success_example['qa'], indent=2))
    
    if failed_count > 0:
        failed_example = next((r for r in results if r["status"] != "success"), None)
        if failed_example:
            print(f"\nExample FAILURE:")
            print(f"Accession: {failed_example['accession_num']}")
            print(f"Raw output (first 200 chars): {failed_example['raw_output'][:200]}")

if __name__ == "__main__":
    main()