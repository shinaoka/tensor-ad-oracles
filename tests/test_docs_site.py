import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class DocsSiteContractTests(unittest.TestCase):
    def test_docs_site_contract_files_exist(self) -> None:
        self.assertTrue((REPO_ROOT / "docs" / "_quarto.yml").exists())
        self.assertTrue((REPO_ROOT / "docs" / "index.md").exists())
        self.assertTrue((REPO_ROOT / "docs" / "math-registry.md").exists())
        self.assertTrue((REPO_ROOT / ".github" / "workflows" / "docs.yml").exists())
        self.assertTrue((REPO_ROOT / "scripts" / "build_docs_site.sh").exists())


if __name__ == "__main__":
    unittest.main()
