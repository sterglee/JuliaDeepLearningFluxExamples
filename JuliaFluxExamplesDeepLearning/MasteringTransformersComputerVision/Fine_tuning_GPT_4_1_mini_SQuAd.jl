using HTTP, JSON3, Base64

# 1. Configuration
const API_KEY = get(ENV, "OPENAI_API_KEY", "YOUR_KEY_HERE")
const AUTH_HEADER = ["Authorization" => "Bearer $API_KEY"]

# --- NEW: Synthetic Data Generator ---
# This creates a training file with SQuAD-style Q&A pairs
function generate_synthetic_data(filename="synthetic_squad.jsonl")
    println("--- Step 0: Generating Synthetic Data ---")
    
    # Synthetic examples following the OpenAI Chat format
    examples = [
        Dict("messages" => [
            Dict("role" => "system", "content" => "You are a reading comprehension assistant."),
            Dict("role" => "user", "content" => "Context: Julia is a high-performance language. Question: Is Julia fast?"),
            Dict("role" => "assistant", "content" => "Yes, Julia is a high-performance language.")
        ]),
        Dict("messages" => [
            Dict("role" => "system", "content" => "You are a reading comprehension assistant."),
            Dict("role" => "user", "content" => "Context: GPT-4o-mini was released in 2024. Question: When was GPT-4o-mini released?"),
            Dict("role" => "assistant", "content" => "GPT-4o-mini was released in 2024.")
        ]),
        Dict("messages" => [
            Dict("role" => "system", "content" => "You are a reading comprehension assistant."),
            Dict("role" => "user", "content" => "Context: The API supports JSONL files. Question: What file format is supported?"),
            Dict("role" => "assistant", "content" => "The API supports JSONL files.")
        ])
    ]

    open(filename, "w") do f
        for ex in examples
            println(f, JSON3.write(ex))
        end
    end
    println("File $filename generated with $(length(examples)) examples.")
    return filename
end

# 2. Upload Training File
function upload_training_file(file_path)
    println("--- Step 1: Uploading Dataset ---")
    body = HTTP.Form([
        "purpose" => "fine-tune",
        "file" => HTTP.Multipart(basename(file_path), open(file_path), "application/jsonl")
    ])

    resp = HTTP.post("https://api.openai.com/v1/files", AUTH_HEADER, body)
    file_info = JSON3.read(resp.body)
    println("File Uploaded. ID: ", file_info.id)
    return file_info.id
end

# 3. Create Fine-Tuning Job
function create_tuning_job(file_id)
    println("\n--- Step 2: Creating Fine-tuning Job ---")
    body = Dict(
        "training_file" => file_id,
        "model" => "gpt-4o-mini-2024-07-18"
    )

    resp = HTTP.post("https://api.openai.com/v1/fine_tuning/jobs",
                     vcat(AUTH_HEADER, ["Content-Type" => "application/json"]),
                     JSON3.write(body))

    job_info = JSON3.read(resp.body)
    println("Job Created. ID: ", job_info.id)
    return job_info.id
end

# 4. Monitor Job Status
function check_job_status(job_id)
    resp = HTTP.get("https://api.openai.com/v1/fine_tuning/jobs/$job_id", AUTH_HEADER)
    status_info = JSON3.read(resp.body)
    println("Current Status: ", status_info.status)
    return status_info.status
end

# --- FULL EXECUTION FLOW ---
# 1. Generate the local file
synthetic_file = generate_synthetic_data()

# 2. Upload to OpenAI
file_id = upload_training_file(synthetic_file)

# 3. Trigger the fine-tune
job_id = create_tuning_job(file_id)

# 4. Check initial status
check_job_status(job_id)

