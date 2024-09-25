import subprocess
import sys


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
    if len(sys.argv) != 2:
        print("Usage: python script.py <version>")
        sys.exit(1)

    version = sys.argv[1]
    confirm = input(f"Deploying version {version}. Would you like to continue? (y/n)")
    if confirm.lower() != "y":
        print("Deployment cancelled.")
        return
    if "v" in version:
        raise ValueError("Version should not contain 'v'.")
    create_git_tag(version)
    build_package()
    publish_package(version)


if __name__ == "__main__":
    main()
