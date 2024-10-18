import subprocess
import sys
from argparse import ArgumentParser


def create_git_tag(version):
    tag_name = f"v{version}"
    result = subprocess.run(["git", "tag", tag_name])
    if result.returncode != 0:
        raise RuntimeError("Failed to create git tag.")
    result = subprocess.run(["git", "push", "origin", tag_name])
    if result.returncode != 0:
        raise RuntimeError("Failed to push git tag.")
    else:
        print(f"Git tag {tag_name} created and pushed.")


def build_package():
    res = subprocess.run(["python", "-m", "build"])
    if res.returncode != 0:
        raise RuntimeError("Failed to build package.")
    print("Package built successfully.")


def publish_package(version):
    res = subprocess.run(["twine", "upload", f"dist/*{version}*"])
    if res.returncode != 0:
        raise RuntimeError("Failed to publish package.")
    print("Package published successfully.")


def main():
    parser = ArgumentParser()
    parser.add_argument("version", type=str, help="The version to deploy")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Only publish the package to PyPI, do not create a git tag",
        default=False,
    )
    args = parser.parse_args()
    version = args.version
    res = subprocess.run(["git", "tag", "-l", "v*"], capture_output=True, text=True)
    last_tag = res.stdout.strip().split("\n")[-1]
    print(f"Last tag: {last_tag}")
    confirm = input(f"Deploying version {version}. Would you like to continue? (y/n)")
    if confirm.lower() != "y":
        print("Deployment cancelled.")
        return
    if "v" in version:
        raise ValueError("Version should not contain 'v'.")

    if not args.publish:
        if last_tag == "v" + version:
            overwrite = input(
                "Git tag already exists, would you like to overwrite it? (y/n) "
            )
            if overwrite.lower() != "y":
                print("Deployment cancelled.")
                return
            else:
                print("Overwriting git tag.")
                subprocess.run(["git", "tag", "-d", last_tag])
                subprocess.run(["git", "push", "origin", "tag", "-d", last_tag])
                print("Git tag overwritten.")
        create_git_tag(version)
        build_package()
    publish_package(version)


if __name__ == "__main__":
    main()
