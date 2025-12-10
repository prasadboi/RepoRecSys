import requests
import base64

GITHUB_TOKEN = "token"   # recommended to avoid rate limits
headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}

def get_repo_from_id(repo_id):
    url = f"https://api.github.com/repositories/{repo_id}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()

def get_readme(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    r = requests.get(url, headers=headers)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    return base64.b64decode(data["content"]).decode("utf-8")

repo_info = get_repo_from_id(51268244)
owner = repo_info["owner"]["login"]
repo_name = repo_info["name"]

readme_text = get_readme(owner, repo_name)
print(readme_text)
