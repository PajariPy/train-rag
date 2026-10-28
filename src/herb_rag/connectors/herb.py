"""Connector for the HERB benchmark stored as local JSON files."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter

from herb_rag.connectors.base import DataConnector
from herb_rag.connectors.herb_mapping import product_to_artifacts, product_to_questions
from herb_rag.models import Corpus, Customer, Employee
from herb_rag.models.herb_raw import RawProduct

EMPLOYEES_ADAPTER = TypeAdapter(dict[str, Employee])
CUSTOMERS_ADAPTER = TypeAdapter(list[Customer])


class HerbConnector(DataConnector):
    """Load HERB from a local directory laid out like the Hugging Face dataset.

    Expected layout: `metadata/{employee,customers_data}.json` and one
    `products/<Product>.json` file per product.
    """

    name = "herb"

    def __init__(self, data_dir: Path, products: Sequence[str] | None = None) -> None:
        """Configure the connector.

        Args:
            data_dir: Root of the downloaded HERB dataset.
            products: Product names to load; all products when None.
        """
        self.data_dir = data_dir
        self.products = products

    def load(self) -> Corpus:
        """Load artifacts, questions, employees and customers.

        Returns:
            The normalized HERB corpus.

        Raises:
            FileNotFoundError: If the data directory or a requested product is missing.
        """
        corpus = Corpus(
            source=self.name,
            employees=self.load_employees(),
            customers=self.load_customers(),
        )
        for path in self.product_paths():
            raw = RawProduct.model_validate_json(path.read_bytes())
            corpus.artifacts += product_to_artifacts(path.stem, raw)
            corpus.questions += product_to_questions(path.stem, raw)

        return corpus

    def product_paths(self) -> list[Path]:
        """Resolve the product files to load.

        Returns:
            Paths of the product JSON files, sorted by product name.

        Raises:
            FileNotFoundError: If the products folder or a requested product is missing.
        """
        products_dir = self.data_dir / "products"
        if not products_dir.is_dir():
            raise FileNotFoundError(
                f"No HERB products at {products_dir}; run `uv run herb-rag-download` first."
            )

        if self.products is None:
            return sorted(products_dir.glob("*.json"))

        paths = [products_dir / f"{product}.json" for product in self.products]
        missing = [path.stem for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Unknown HERB products: {', '.join(missing)}")

        return paths

    def load_employees(self) -> list[Employee]:
        """Load the employee directory.

        Returns:
            All employees of the simulated company.
        """
        raw = (self.data_dir / "metadata" / "employee.json").read_bytes()

        return list(EMPLOYEES_ADAPTER.validate_json(raw).values())

    def load_customers(self) -> list[Customer]:
        """Load the customer directory.

        Returns:
            All customer contacts.
        """
        raw = (self.data_dir / "metadata" / "customers_data.json").read_bytes()

        return CUSTOMERS_ADAPTER.validate_json(raw)
