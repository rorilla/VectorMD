# VectorMD

`VectorMD` enables users to convert markdown documents into a semantic searchable format. By embedding markdown headings into vector space, it can quickly find relevant sections in markdown documents based on natural language queries.

## Features

- **Semantic Search**: Converts markdown documents into a semantically searchable database.
- **Efficient Indexing**: Employs FAISS for lightning-fast vector searches.
- **Easy Integration**: Works seamlessly in both CLI environments and directly within Python.

## Installation

Install `VectorMD` via pip:

```bash
pip install VectorMD
```

## Quick Start

### Command Line Interface (CLI)

1. **Initialization**:

   Convert your markdown document into a semantically searchable format using the following command:
   
   ```bash
   vmd-init --file path_to_code_snippets.md
   ```

2. **Querying**:

   After initialization, search for relevant sections in your document using:
   
   ```bash
   vmd docker compose quantized llama2
   ```

### Python

Use VectorMD directly within your Python scripts:

```python
from vectormd import VectorMD

# Initialize with your markdown file
medicalDB = VectorMD("path_to_medical_markdown.md")

# Query
results = medicalDB.query("HACP empiric tx regimen duration")
```

## Contributing

We welcome contributions! If you'd like to help improve `VectorMD`, please fork the repository and submit a pull request.

## Contribution

Contributions are always welcome! If you'd like to help improve VectorMD, please fork the repository and submit a pull request.

## License

[Apache-2.0 license](LICENSE)
