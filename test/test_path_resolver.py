import os
import pytest
from researchgraph.test_utils.path_resolver import TestPathResolver


def test_path_resolver_initialization():
    """Test that the path resolver initializes correctly and creates the save directory."""
    resolver = TestPathResolver()
    assert os.path.exists(resolver.get_save_dir())
    assert "test/outputs" in resolver.get_save_dir()


def test_test_file_path_resolution():
    """Test that test file paths are correctly constructed with subdirectories."""
    resolver = TestPathResolver()
    path = resolver.get_test_file_path("test.txt", subdir="data")
    assert os.path.dirname(path).endswith("data")
    assert path.endswith("test.txt")

    # Test without subdir
    simple_path = resolver.get_test_file_path("test.txt")
    assert simple_path.endswith("test.txt")
    assert "data" not in os.path.dirname(simple_path)


def test_template_dir_resolution():
    """Test that template directory paths are correctly constructed."""
    resolver = TestPathResolver()
    template_path = resolver.get_template_dir("2d_diffusion")
    assert os.path.join("templates", "2d_diffusion") in template_path


def test_figures_dir_resolution():
    """Test that figures directory path is correctly constructed."""
    resolver = TestPathResolver()
    figures_path = resolver.get_figures_dir()
    assert figures_path.endswith("images")
