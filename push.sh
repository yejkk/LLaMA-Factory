git ls-files > file_list.txt
rsync -avz --files-from=file_list.txt ./ ai_server:/opt/services/yx_test/llama-factory-new-grpo