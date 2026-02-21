import requests

def get_jobs_for_role(role, max_results=5):
    url = "https://jsearch.p.rapidapi.com/search"
    querystring = {"query": role, "num_pages": "1"}
    headers = {
        "X-RapidAPI-Key": "d7c16a9efdmsh3f85b85a7b8ae51p14e8cfjsnfda220ee5c6f",  # Replace with your key
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"

    }

    try:
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code != 200:
            return [("Error fetching jobs (status {})".format(response.status_code), "#")]

        data = response.json()
        jobs = []

        for job in data.get("data", [])[:max_results]:
            title = job.get("job_title", "No Title")
            link = job.get("job_apply_link") or job.get("job_google_link") or "#"
            jobs.append((title, link))

        if not jobs:
            return [("No jobs found", "#")]

        return jobs
    except Exception as e:
        return [(f"Error fetching jobs: {e}", "#")]
