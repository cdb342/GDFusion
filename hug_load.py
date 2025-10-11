from huggingface_hub import HfApi
api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_folder(
    folder_path="/home/yc37439/code/alocc2/work_dir2/gdfusion/",
    repo_id="Dobbin/OccStudio",
    repo_type="model",
)
api.upload_folder(
    folder_path="/home/yc37439/code/alocc2/work_dir2/causalocc/",
    repo_id="Dobbin/OccStudio",
    repo_type="model",
)
api.upload_folder(
    folder_path="/home/yc37439/code/alocc2/work_dir2/fbocc/",
    repo_id="Dobbin/OccStudio",
    repo_type="model",
)
api.upload_folder(
    folder_path="/home/yc37439/code/alocc2/work_dir2/bevdetocc/",
    repo_id="Dobbin/OccStudio",
    repo_type="model",
)