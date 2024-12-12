import pytest
import os
import json
from researchgraph.graphs.ai_scientist.ai_scientist_node.execute_idea import IdeaExecutionComponent
from researchgraph.graphs.ai_scientist.ai_scientist_node.perform_experiments import ExperimentComponent
from researchgraph.graphs.ai_scientist.ai_scientist_node.generate_ideas import IdeaGenerationComponent

def test_idea_execution():
    # Test idea execution with test domain
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'researchgraph', 'graphs', 'ai_scientist', 'templates', 'test_domain')
    executor = IdeaExecutionComponent()

    # Load test ideas
    with open(os.path.join(base_dir, 'seed_ideas.json'), 'r') as f:
        ideas = json.load(f)

    # Test execution of first idea
    result = executor.execute(ideas[0], base_dir)
    assert result is not None
    assert 'final_info.json' in os.listdir(os.path.join(base_dir, 'run_0'))

def test_experiment_component():
    # Test experiment execution
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'researchgraph', 'graphs', 'ai_scientist', 'templates', 'test_domain')
    experiment = ExperimentComponent()

    # Test running experiment
    result = experiment.run_experiment(base_dir)
    assert result is not None
    assert os.path.exists(os.path.join(base_dir, 'run_0', 'final_info.json'))

    # Verify results format
    with open(os.path.join(base_dir, 'run_0', 'final_info.json'), 'r') as f:
        data = json.load(f)
    assert 'mlp' in data
    assert 'cnn' in data
    assert 'test_accuracy' in data['mlp']
    assert 'train_accuracy' in data['mlp']
    assert 'training_time' in data['mlp']

def test_idea_generation():
    # Test idea generation
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'researchgraph', 'graphs', 'ai_scientist', 'templates', 'test_domain')
    generator = IdeaGenerationComponent()

    # Load seed ideas
    with open(os.path.join(base_dir, 'seed_ideas.json'), 'r') as f:
        seed_ideas = json.load(f)

    # Test idea generation
    ideas = generator.generate(seed_ideas, base_dir)
    assert isinstance(ideas, list)
    assert len(ideas) > 0
    for idea in ideas:
        assert 'Name' in idea
        assert 'Title' in idea
        assert 'Experiment' in idea
