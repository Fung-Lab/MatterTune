import os

for dir in os.listdir('./'):
    if dir.startswith("checkpoints"):
        files = os.listdir(f'./{dir}')
        for file in files:
            if file.endswith(".ckpt"):
                os.system(f'mv ./{dir}/{file} /storage/lingyu/checkpoints/{dir}/')