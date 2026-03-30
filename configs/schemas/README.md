# Using type hinting for experiments' configurations
Hi 👋, this directory stores the schemas to build fully functional YAML experiments.
To use them, install [redhat-yaml](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml) and add the following to your VS Code `settings.json`:
```
"yaml.schemas": {
    "configs/schemas/experiment.json": [
        "**/*.exp.yml",
        "**/*.exp.yaml"
    ],
    "configs/schemas/experiment_with_corruption.json": [
        "**/*.cexp.yml",
        "**/*.cexp.yaml"
    ]
}
```
