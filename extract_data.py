import ast
import os
import pathlib
import re
import sys
from fnmatch import fnmatch
from typing import Union
import pandas as pd

gathered_data = []
# Remove private names and main functions
is_ok = lambda name: (name != "main") and not name.startswith("_") and ("test" not in name.lower()) and ("error" not in name.lower())
is_method = (
    lambda func: len(func.args.args) > 0
    and func.args
    and "self" in func.args.args[0].arg
)

class CustomVisitor(ast.NodeVisitor):
    """
    Custom NodeVisitor to extract functions
    methods and classes with comments.
    """
    def __init__(self, filepath: pathlib.Path, node) -> None:
        super().__init__()
        self.filepath = filepath
        parsed = ast.parse(node)
        self.visit(parsed)
    def add_data(self, node: Union[ast.FunctionDef, ast.ClassDef], type_):
        """
        Load data
        """
        comment = ast.get_docstring(node)
        if comment:
            comment = comment.split("\n")[0]
            comment = re.sub(r"\[\]\{\}\;\'\"\>\?\<\\:\!\#\@\%\^\*\(\)", r"", comment)
            comment = re.sub(r"\s+", r" ", comment)
        data = {
            "name": node.name,
            "path": self.filepath,
            "comment": comment,
            "line": node.lineno,
            "type": type_,
        }
        gathered_data.append(data)

    def visit_ClassDef(self, node: ast.ClassDef):
        """
        Visit classes
        """
        self.generic_visit(node)
        if is_ok(node.name):
            self.add_data(node, "class")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Visit functions and methods
        """
        if is_ok(node.name):
            self.add_data(node, "method" if is_method(node) else "function")

def find_files(directory):
    """
    Get all files
    """
    i = 0
    for path, _, files in os.walk(directory):
        for file in files:
            if fnmatch(file, "*.py"):
                file_path = os.path.join(path, file)
                i += 1
                with open(file_path) as file:
                    CustomVisitor(file_path, file.read())
    print(f"Python files  {i}")
    dataframe = pd.DataFrame(gathered_data)
    dataframe.to_csv("data/data.csv", index=False, encoding="utf-8")
    try:
        print(dataframe["type"].value_counts())
    except KeyError:
        print("No python file was found in the source directory.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: python extract_data.py directory_path")
        sys.exit(2)
    find_files(sys.argv[1])
