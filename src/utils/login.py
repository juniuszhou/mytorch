from huggingface_hub import login


def read_env():
    with open("../.env", "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.strip().split("=")
            if key == "HF_KEY":
                return value


def login_huggingface():
    token = read_env()
    login(new_session=True, token=token)
