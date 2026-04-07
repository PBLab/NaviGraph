# Configure matplotlib backend before any other imports
import os
import matplotlib
import warnings

# Fix QT_API environment variable if needed
if os.environ.get('QT_API') == 'pyqt':
    os.environ['QT_API'] = 'pyqt5'

# Setup matplotlib backend with fallback options
def setup_matplotlib_backend():
    """Setup matplotlib backend for GUI functionality."""
    backends_to_try = ['Qt5Agg', 'TkAgg', 'Qt4Agg', 'GTK3Agg']
    
    for backend in backends_to_try:
        try:
            # Suppress warnings during backend testing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                matplotlib.use(backend, force=True)
                
                # Try to import and test the backend
                import matplotlib.pyplot as plt
                fig = plt.figure()
                plt.close(fig)
                return backend
        except Exception:
            continue
    
    # If no GUI backend works, use non-interactive backend
    matplotlib.use('Agg', force=True)
    return 'Agg'

# Setup backend before any matplotlib imports
backend = setup_matplotlib_backend()

"""
NaviGraph CLI - Command-line interface for spatial navigation analysis.

This module provides the command-line interface for NaviGraph, supporting
experiment running, configuration validation, and interactive setup tools.
"""

import sys
import os
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import click
from omegaconf import OmegaConf

# Import experiment runner and enums
from ..core.experiment_runner import ExperimentRunner
from ..core.enums import SystemMode


def resolve_project_root(config_path: Path) -> Path:
    """Find the project root directory from a config path.
    
    Searches upward from the config file location to find the project root,
    identified by the presence of a .git directory or pyproject.toml file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Path to the project root directory
    """
    current = config_path.parent.absolute()
    
    # Search upward for project root indicators
    while current != current.parent:
        if (current / '.git').exists() or (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    
    # If no project root found, use config directory
    return config_path.parent.absolute()


def process_config_path(config_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process configuration with proper path resolution.
    
    Updates the configuration with resolved paths based on config file location,
    adding special handling for {PROJECT_ROOT} placeholder in paths.
    
    Args:
        config_path: Path to the configuration file  
        config: Loaded configuration dictionary
        
    Returns:
        Updated configuration with resolved paths
    """
    config_dir = config_path.parent.absolute()
    project_root = resolve_project_root(config_path)
    
    # Add metadata
    config['_config_path'] = str(config_path)
    config['_config_dir'] = str(config_dir)
    config['_project_root'] = str(project_root)
    
    # Process experiment_output_path with {PROJECT_ROOT} support
    if 'experiment_output_path' in config:
        output_path = config['experiment_output_path']
        
        if isinstance(output_path, str):
            # Replace {PROJECT_ROOT} placeholder
            if '{PROJECT_ROOT}' in output_path:
                output_path = output_path.replace('{PROJECT_ROOT}', str(project_root))
            
            # Resolve path
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = config_dir / output_path
            
            config['experiment_output_path'] = str(output_path.resolve())
    
    # Process experiment_path - should be relative to config file location
    if 'experiment_path' in config:
        exp_path = config['experiment_path']
        if isinstance(exp_path, str):
            exp_path = Path(exp_path)
            if not exp_path.is_absolute():
                exp_path = config_dir / exp_path
            config['experiment_path'] = str(exp_path.resolve())
    
    return config


@click.group()
@click.version_option(version='0.2.0', prog_name='NaviGraph')
def cli():
    """NaviGraph - A flexible framework for spatial navigation analysis.
    
    NaviGraph provides tools for analyzing animal navigation behavior through
    integration of pose estimation, spatial mapping, and graph-based analysis.
    
    \b
    Common Commands:
      navigraph run config.yaml              - Run analysis and visualization
      navigraph run analyze config.yaml      - Run analysis only  
      navigraph run visualize config.yaml    - Run visualization only
      navigraph setup graph config.yaml      - Setup graph mapping
      navigraph setup calibration config.yaml - Setup camera calibration
      navigraph validate config.yaml         - Validate configuration
    
    Use 'navigraph COMMAND --help' for more information on each command.
    """
    pass


@cli.command()
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--show', is_flag=True, help='Show visualization results after creation')
@click.option('--analyze', is_flag=True, help='Run analysis only')
@click.option('--visualize', is_flag=True, help='Run visualization only')
def run(config_path: Path, verbose: bool, show: bool, analyze: bool, visualize: bool):
    """Run NaviGraph experiments with different execution modes.
    
    CONFIG_PATH: Path to YAML configuration file
    
    \b
    Examples:
      navigraph run config.yaml              - Run both analysis and visualization
      navigraph run config.yaml --show       - Run and show visualization results
      navigraph run config.yaml --analyze    - Run analysis only
      navigraph run config.yaml --visualize  - Run visualization only
      navigraph run config.yaml --analyze --visualize  - Run both explicitly
    """
    # Determine modes based on flags
    if analyze and visualize:
        modes = [SystemMode.ANALYZE, SystemMode.VISUALIZE]
    elif analyze:
        modes = [SystemMode.ANALYZE]
    elif visualize:
        modes = [SystemMode.VISUALIZE]
    else:
        # Default: run both analysis and visualization
        modes = [SystemMode.ANALYZE, SystemMode.VISUALIZE]
    
    _run_experiment_with_modes(config_path, verbose, modes, show)


def _run_experiment_with_modes(config_path: Path, verbose: bool, modes: List[SystemMode], show: bool = False):
    """Helper function to run experiment with specified modes."""
    try:
        modes_str = ', '.join(mode.value for mode in modes)
        click.echo(f"🚀 Starting NaviGraph experiment ({modes_str})")
        click.echo(f"📋 Configuration: {config_path}")
        
        # Load and process configuration
        config = OmegaConf.load(config_path)
        config = process_config_path(config_path, OmegaConf.to_container(config))
        
        # Override verbose if specified
        if verbose:
            config['verbose'] = True
        
        # Add show flag to config for visualization system
        if show:
            config['show_visualization'] = True
            
        # Create and run experiment with specified modes
        runner = ExperimentRunner(config, system_modes=modes)
        
        results = runner.run_experiment()
        
        if results is not None:
            click.echo(f"✅ Experiment completed successfully!")
            click.echo(f"📊 Analysis results: {len(results.columns)} sessions, {len(results.index)} metrics")
            
            output_path = config.get('experiment_output_path', '.')
            click.echo(f"📁 Results saved to: {output_path}")
        else:
            click.echo("✅ Experiment completed (no analysis results)")
            
    except Exception as e:
        click.echo(f"❌ Experiment failed: {str(e)}", err=True)
        if config.get('verbose', False):
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)






@cli.command()
@click.argument('data_path', type=click.Path(exists=True, path_type=Path))
@click.option('--config', '-c', type=click.Path(exists=True, path_type=Path),
              help='Configuration file to use for validation (optional)')
@click.option('--format', '-f', 
              type=click.Choice(['table', 'json', 'tree']),
              default='table',
              help='Output format for discovered sessions')
def discover(data_path: Path, config: Optional[Path], format: str):
    """Discover and list available experimental sessions with validation.
    
    Scan a directory for sessions and validate them against configured data sources.
    Without a config file, performs basic directory scanning.
    
    DATA_PATH: Path to directory containing experimental data
    
    \b
    Examples:
      navigraph discover ./data/
      navigraph discover ./data/ --config config.yaml --format json
      navigraph discover ./sessions/ --format tree
    
    \b
    With config file:
      • Validates sessions using configured data source plugins
      • Shows detailed validation results per session
      • Respects plugin file patterns and requirements
    
    \b
    Without config file:
      • Basic directory scanning for potential sessions
      • Lists directories that could contain session data
    """
    # Load plugins for session discovery
    from .. import plugins  # noqa: F401
    
    try:
        click.echo(f"🔍 Discovering sessions in: {data_path}")
        
        if config:
            # Use provided config for plugin-based validation
            click.echo(f"📋 Using configuration: {config}")
            
            # Load configuration
            config_data = OmegaConf.load(config)
            config_data = process_config_path(config, OmegaConf.to_container(config_data))
            
            # Override experiment path to discovery location
            config_data.experiment_path = str(data_path)
            
            # Create experiment runner
            runner = ExperimentRunner(config_data)
            
            # Discover sessions
            sessions = runner.discover_sessions()
            
            if not sessions:
                click.echo("⚠️  No session directories found")
                return
            
            click.echo(f"📁 Found {len(sessions)} session directories")
            
            # Validate sessions using plugins
            validation_report = runner.validate_sessions(sessions)
            
            # Display results
            if format == 'table':
                click.echo()
                click.echo("Session Validation Results:")
                click.echo("=" * 60)
                
                for session_validation in validation_report.session_validations:
                    status = "✅ VALID" if session_validation.is_valid else "❌ INVALID"
                    click.echo(f"\n{session_validation.session_id}: {status}")
                    
                    # Show data source results
                    for ds_result in session_validation.data_source_results:
                        icon = "✓" if ds_result.is_valid else "✗"
                        click.echo(f"  {icon} {ds_result.name}: ", end="")
                        
                        if ds_result.is_valid:
                            click.echo(f"{len(ds_result.files)} files")
                            if config_data.get('verbose'):
                                for file in ds_result.files[:3]:  # Show first 3
                                    click.echo(f"      • {file.name}")
                                if len(ds_result.files) > 3:
                                    click.echo(f"      ... and {len(ds_result.files)-3} more")
                        else:
                            click.echo(f"{ds_result.error or 'No matching files'}")
                
                # Summary
                click.echo()
                click.echo("Summary:")
                click.echo(f"  Valid sessions: {validation_report.valid_count}/{validation_report.total_count}")
                
                if validation_report.warnings:
                    click.echo(f"  ⚠️  Warnings: {len(validation_report.warnings)}")
                    for warning in validation_report.warnings[:3]:
                        click.echo(f"     • {warning}")
                        
            elif format == 'json':
                import json
                output = {
                    'total_sessions': validation_report.total_count,
                    'valid_sessions': validation_report.valid_count,
                    'sessions': []
                }
                
                for sv in validation_report.session_validations:
                    session_data = {
                        'id': sv.session_id,
                        'valid': sv.is_valid,
                        'data_sources': {}
                    }
                    
                    for ds in sv.data_source_results:
                        session_data['data_sources'][ds.name] = {
                            'valid': ds.is_valid,
                            'files': len(ds.files) if ds.is_valid else 0,
                            'error': ds.error
                        }
                    
                    output['sessions'].append(session_data)
                
                click.echo(json.dumps(output, indent=2))
                
            elif format == 'tree':
                # Tree-like display
                for i, sv in enumerate(validation_report.session_validations):
                    is_last = i == len(validation_report.session_validations) - 1
                    prefix = "└── " if is_last else "├── "
                    status = "✅" if sv.is_valid else "❌"
                    click.echo(f"{prefix}{sv.session_id} {status}")
                    
                    for j, ds in enumerate(sv.data_source_results):
                        is_last_ds = j == len(sv.data_source_results) - 1
                        tree_prefix = "    " if is_last else "│   "
                        ds_prefix = "└── " if is_last_ds else "├── "
                        icon = "✓" if ds.is_valid else "✗"
                        files_info = f"({len(ds.files)} files)" if ds.is_valid else "(no files)"
                        click.echo(f"{tree_prefix}{ds_prefix}{icon} {ds.name} {files_info}")
        else:
            # Basic directory scanning without config
            click.echo("📂 Basic directory scan (no config provided)")
            
            # Find all subdirectories
            subdirs = [d for d in data_path.iterdir() if d.is_dir()]
            
            if not subdirs:
                click.echo("⚠️  No subdirectories found")
                return
            
            click.echo(f"📁 Found {len(subdirs)} potential session directories:")
            
            for subdir in sorted(subdirs):
                # Count files in directory
                file_count = len(list(subdir.glob('*')))
                click.echo(f"  • {subdir.name} ({file_count} items)")
                
    except Exception as e:
        click.echo(f"❌ Discovery failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command('list-graph-builders')
def list_graph_builders_cmd():
    """List all available graph builders.
    
    Display registered graph builders that can be used in configuration files.
    
    \b
    Example:
      navigraph list-graph-builders
    """
    try:
        from ..core.graph.builders import list_graph_builders, get_graph_builder_info
        
        builders = list_graph_builders()
        
        if not builders:
            click.echo("No graph builders registered")
            return
        
        click.echo("📊 Available Graph Builders")
        click.echo("-" * 40)
        
        for builder_name in sorted(builders):
            info = get_graph_builder_info(builder_name)
            click.echo(f"\n• {builder_name}")
            click.echo(f"  Class: {info['class_name']}")
            
            if info['docstring']:
                # Get first line of docstring
                doc_lines = info['docstring'].strip().split('\n')
                if doc_lines:
                    click.echo(f"  Description: {doc_lines[0]}")
            
            if info['parameters']:
                click.echo("  Parameters:")
                for param_name, param_info in info['parameters'].items():
                    required = "required" if param_info['required'] else "optional"
                    default = f", default={param_info['default']}" if param_info['default'] is not None else ""
                    click.echo(f"    - {param_name} ({required}{default})")
        
        click.echo(f"\nTotal: {len(builders)} builders")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command('list-plugins')
@click.option('--category', '-c',
              type=click.Choice(['plugins', 'session_metrics', 'cross_session_metrics', 'visualizers', 'all']),
              default='all',
              help='Plugin category to list')
@click.option('--format', '-f',
              type=click.Choice(['table', 'json', 'simple']),
              default='table',
              help='Output format')
def list_plugins(category: str, format: str):
    """List all available NaviGraph plugins by category.
    
    Display plugins for data integration, analysis, and visualization.
    Use this to discover available functionality and verify plugin registration.
    
    \b
    Examples:
      navigraph list-plugins
      navigraph list-plugins --category analyzers
      navigraph list-plugins --format json
    
    \b
    Plugin categories:
      plugins           - All unified NaviGraph plugins (pose tracking, calibration, etc.)
      session_metrics   - Session-level analysis functions
      cross_session_metrics - Cross-session analysis functions
      visualizers       - Visualization functions
    """
    try:
        click.echo("🔌 NaviGraph Plugin Registry")
        
        # Load plugins to populate the registry
        from .. import plugins
        from ..core.registry import registry
        
        # Get all plugins
        all_plugins = registry.list_all_plugins()
        
        # Filter by category if specified
        if category != 'all':
            all_plugins = {category: all_plugins.get(category, [])}
        
        # Remove empty categories
        all_plugins = {k: v for k, v in all_plugins.items() if v}
        
        if not all_plugins:
            click.echo("⚠️  No plugins found")
            return
        
        if format == 'simple':
            # Simple list format
            for cat, plugins in all_plugins.items():
                for plugin in plugins:
                    click.echo(f"{cat}:{plugin}")
        
        elif format == 'json':
            # JSON format
            import json
            click.echo(json.dumps(all_plugins, indent=2))
        
        else:  # table format
            # Table format with descriptions
            click.echo()
            
            category_descriptions = {
                'plugins': '📊 Plugins - Unified NaviGraph plugins for data integration',
                'session_metrics': '🧮 Session Metrics - Session-level analysis functions',
                'cross_session_metrics': '📈 Cross-Session Metrics - Cross-session analysis functions',
                'visualizers': '🎨 Visualizers - Visualization functions'
            }
            
            for cat, plugins in all_plugins.items():
                desc = category_descriptions.get(cat, f"📦 {cat.title()}")
                click.echo(f"{desc}")
                click.echo("-" * len(desc))
                
                for plugin in plugins:
                    click.echo(f"  • {plugin}")
                
                click.echo()
            
            total_plugins = sum(len(plugins) for plugins in all_plugins.values())
            click.echo(f"Total: {total_plugins} plugins across {len(all_plugins)} categories")
            
    except Exception as e:
        click.echo(f"❌ Plugin listing failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
def validate(config_path: Path):
    """Validate configuration file for errors and compatibility.
    
    Check configuration syntax, required fields, file paths, and plugin
    dependencies before running experiments to catch issues early.
    
    CONFIG_PATH: Path to YAML configuration file to validate
    
    \b
    Examples:
      navigraph validate examples/basic_maze/config.yaml
      navigraph validate my_experiment_config.yaml
    
    \b
    Validation checks:
      ✓ YAML syntax and structure
      ✓ Required fields present
      ✓ File paths exist and accessible
      ✓ Plugin dependencies satisfied
      ✓ Configuration parameter validity
    """
    try:
        click.echo(f"🔍 Validating configuration: {config_path}")
        
        # Load plugins for validation
        from .. import plugins
        
        # Load configuration
        config = OmegaConf.load(config_path)
        config = process_config_path(config_path, OmegaConf.to_container(config))
        issues = []
        warnings = []
        
        # Check required fields
        required_fields = ['stream_path', 'keypoint_detection_file_path']
        for field in required_fields:
            if field not in config:
                issues.append(f"Missing required field: {field}")
        
        # Check paths exist
        path_fields = ['stream_path', 'keypoint_detection_file_path', 'map_path']
        for field in path_fields:
            if field in config:
                path = Path(config[field])
                if not path.exists():
                    issues.append(f"Path does not exist: {field} = {path}")
        
        # Check plugin configuration
        if 'analyze' in config and 'metrics' in config.analyze:
            metrics = config.analyze.metrics
            from ..core.registry import registry
            available_analyzers = registry.list_all_plugins()['analyzers']
            
            for metric_name, metric_config in metrics.items():
                func_name = metric_config.get('func_name')
                
                # Map function names to analyzers (simplified check)
                analyzer_mapping = {
                    'time_a_to_b': 'spatial_metrics',
                    'velocity_a_to_b': 'spatial_metrics', 
                    'num_nodes_in_path': 'navigation_metrics',
                    'shortest_path_from_a_to_b': 'navigation_metrics',
                    'exploration_percentage': 'exploration_metrics',
                    'avg_node_time': 'exploration_metrics'
                }
                
                required_analyzer = analyzer_mapping.get(func_name)
                if required_analyzer and required_analyzer not in available_analyzers:
                    issues.append(f"Analyzer '{required_analyzer}' required for metric '{metric_name}' not available")
        
        # Report results
        click.echo()
        if issues:
            click.echo("❌ Configuration validation failed:")
            for issue in issues:
                click.echo(f"   • {issue}")
        else:
            click.echo("✅ Configuration is valid!")
        
        if warnings:
            click.echo("⚠️  Warnings:")
            for warning in warnings:
                click.echo(f"   • {warning}")
        
        if issues:
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Validation failed: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
@click.option('--start', '-s', required=True, type=int, help='Starting node ID')
@click.option('--target', '-t', type=int, help='Target node ID (optional)')
@click.option('--n-walks', '-n', default=100, type=int, help='Number of random walks [default: 100]')
@click.option('--max-steps', '-m', type=int, help='Maximum steps per walk (required if no target)')
@click.option('--backtrack-prob', '-b', default=0.0, type=float,
              help='Backtracking probability (-1=uniform, 0.0=none, 1.0=always) [default: 0.0]')
@click.option('--terminate-on-target/--no-terminate-on-target', default=True,
              help='Stop immediately when reaching target [default: True]')
@click.option('--use-weights/--no-weights', default=False,
              help='Use edge weights for transitions [default: False]')
@click.option('--n-jobs', '-j', default=1, type=int,
              help='Number of processes (1=serial, -1=all cores) [default: 1]')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.option('--stats/--no-stats', default=True,
              help='Show summary statistics [default: True]')
@click.option('--save-paths', type=click.Path(path_type=Path),
              help='Save paths to JSON file')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
def walk(config_path: Path, start: int, target: Optional[int], n_walks: int,
         max_steps: Optional[int], backtrack_prob: float, terminate_on_target: bool,
         use_weights: bool, n_jobs: int, seed: Optional[int], stats: bool,
         save_paths: Optional[Path], verbose: bool):
    """Generate random walks on the graph structure from config.

    Performs random walks on a graph defined in the configuration file,
    with flexible control over walk behavior, backtracking, and parallelization.

    CONFIG_PATH: Path to YAML configuration file containing graph definition

    \b
    Examples:
      # Basic: 100 walks of 20 steps from node 0 (no backtracking)
      navigraph walk config.yaml --start 0 --max-steps 20

      # Uniform random walk: backtracking allowed (equal probability)
      navigraph walk config.yaml -s 0 -m 20 -b -1 -n 1000

      # Target-directed: walk from node 0 to node 127
      navigraph walk config.yaml -s 0 -t 127 -m 50 -n 1000

      # With backtracking: 30% chance to return to previous node
      navigraph walk config.yaml -s 0 -m 15 -b 0.3 -n 500

      # Parallel execution on all CPU cores
      navigraph walk config.yaml -s 0 -t 127 -m 50 -n 10000 -j -1

      # Save paths to file
      navigraph walk config.yaml -s 0 -m 20 -n 100 --save-paths walks.json

    \b
    Options explained:
      --start, -s          Starting node for all walks (required)
      --target, -t         Target node (walk ends when reached)
      --n-walks, -n        Number of walks to generate
      --max-steps, -m      Maximum steps per walk
      --backtrack-prob, -b Probability of backtracking (-1=uniform, 0=none, 0-1=explicit)
      --n-jobs, -j         Parallel processes (1=serial, -1=all cores)
      --seed               Random seed for reproducible results
    """
    try:
        # Validate parameters
        if max_steps is None and target is None:
            click.echo("❌ Error: Must provide either --max-steps or --target", err=True)
            sys.exit(1)

        if backtrack_prob != -1 and not (0.0 <= backtrack_prob <= 1.0):
            click.echo(f"❌ Error: backtrack-prob must be -1 or between 0.0 and 1.0, got {backtrack_prob}", err=True)
            sys.exit(1)

        # Load configuration
        if verbose:
            click.echo(f"📋 Loading configuration: {config_path}")

        config = OmegaConf.load(config_path)
        config = process_config_path(config_path, OmegaConf.to_container(config))

        # Import graph modules
        from ..core.graph.structures import GraphStructure
        from ..core.graph.builders import get_graph_builder, list_graph_builders

        # Create graph from config
        graph_section = config.get('graph', {})
        builder_config = graph_section.get('builder', {})

        if not builder_config:
            click.echo("❌ Error: No graph builder configuration found in config file", err=True)
            click.echo("Add 'graph.builder' section to your config file", err=True)
            sys.exit(1)

        graph_type = builder_config.get('type')
        if not graph_type:
            click.echo("❌ Error: No graph builder type specified", err=True)
            click.echo(f"Available builders: {', '.join(list_graph_builders())}", err=True)
            sys.exit(1)

        # Get parameters for the builder
        params = builder_config.get('config', {})

        try:
            # Create graph structure
            graph = GraphStructure.from_config(graph_type, params)

            if verbose:
                click.echo(f"📊 Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
                click.echo(f"   Type: {graph_type}")
                param_str = ', '.join(f"{k}={v}" for k, v in params.items())
                if param_str:
                    click.echo(f"   Parameters: {param_str}")

        except Exception as e:
            click.echo(f"❌ Error creating graph: {str(e)}", err=True)
            sys.exit(1)

        # Validate nodes exist
        if not graph.has_node(start):
            click.echo(f"❌ Error: Start node {start} not in graph", err=True)
            click.echo(f"Available nodes: {sorted(graph.nodes)[:10]}{'...' if graph.num_nodes > 10 else ''}", err=True)
            sys.exit(1)

        if target is not None and not graph.has_node(target):
            click.echo(f"❌ Error: Target node {target} not in graph", err=True)
            click.echo(f"Available nodes: {sorted(graph.nodes)[:10]}{'...' if graph.num_nodes > 10 else ''}", err=True)
            sys.exit(1)

        # Build parameters display
        params_display = []
        params_display.append(f"Start node: {start}")
        if target is not None:
            params_display.append(f"Target node: {target}")
        params_display.append(f"Walks: {n_walks}")
        if max_steps is not None:
            params_display.append(f"Max steps: {max_steps}")
        params_display.append(f"Backtrack prob: {backtrack_prob:.2f}")
        if n_jobs != 1:
            params_display.append(f"Parallel: {n_jobs if n_jobs > 0 else 'all cores'}")
        if seed is not None:
            params_display.append(f"Seed: {seed}")

        click.echo(f"🚶 Running random walks...")
        click.echo(f"   {', '.join(params_display)}")

        # Measure execution time
        import time
        start_time = time.time()

        # Run random walks
        result = graph.random_walks(
            start_node=start,
            target_node=target,
            n_walks=n_walks,
            max_steps=max_steps,
            terminate_on_target=terminate_on_target,
            backtrack_prob=backtrack_prob,
            use_edge_weights=use_weights,
            return_stats=stats,
            seed=seed,
            n_jobs=n_jobs
        )

        execution_time = time.time() - start_time

        # Extract paths and stats
        if stats:
            paths, walk_stats = result
        else:
            paths = result
            walk_stats = None

        # Display results
        click.echo(f"✅ Completed in {execution_time:.2f}s")
        click.echo()

        if walk_stats:
            click.echo("📊 Summary Statistics:")
            click.echo(f"   Mean path length: {walk_stats['mean_length']:.2f} steps")
            click.echo(f"   Median path length: {walk_stats['median_length']:.1f} steps")
            click.echo(f"   Std deviation: {walk_stats['std_length']:.2f} steps")
            click.echo(f"   Min-Max: {walk_stats['min_length']}-{walk_stats['max_length']} steps")

            if target is not None:
                click.echo(f"   Success rate: {walk_stats['success_rate']:.1%}")
                click.echo(f"   Successful walks: {len(walk_stats['successful_walks'])}/{n_walks}")

                # Calculate efficiency vs shortest path
                try:
                    shortest = graph.get_shortest_path(start, target)
                    if shortest:
                        shortest_len = len(shortest) - 1
                        efficiency = shortest_len / walk_stats['mean_length'] if walk_stats['mean_length'] > 0 else 0
                        click.echo(f"   Shortest path: {shortest_len} steps")
                        click.echo(f"   Efficiency: {efficiency:.1%}")
                except:
                    pass

        if verbose and paths:
            click.echo()
            click.echo("📝 First 3 walks:")
            for i, path in enumerate(paths[:3]):
                path_str = ' → '.join(str(n) for n in path[:10])
                if len(path) > 10:
                    path_str += ' → ...'
                click.echo(f"   Walk {i+1}: {path_str} ({len(path)-1} steps)")

        # Save paths if requested
        if save_paths:
            try:
                import json
                output_data = {
                    'parameters': {
                        'start_node': start,
                        'target_node': target,
                        'n_walks': n_walks,
                        'max_steps': max_steps,
                        'backtrack_prob': backtrack_prob,
                        'terminate_on_target': terminate_on_target,
                        'use_weights': use_weights,
                        'seed': seed
                    },
                    'paths': paths,
                    'statistics': walk_stats if walk_stats else {}
                }

                with open(save_paths, 'w') as f:
                    json.dump(output_data, f, indent=2)

                click.echo()
                click.echo(f"💾 Paths saved to: {save_paths}")

            except Exception as e:
                click.echo(f"⚠️  Warning: Failed to save paths: {str(e)}", err=True)

    except Exception as e:
        click.echo(f"❌ Random walk failed: {str(e)}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


# Graph setup and testing commands
@cli.group()
def setup():
    """Setup and configuration tools for NaviGraph components.
    
    Interactive tools for configuring graph mappings, calibration,
    and other system components.
    
    \b
    Examples:
      navigraph setup graph config.yaml
      navigraph setup calibration config.yaml
    """
    pass


@setup.command('graph')
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
def setup_graph(config_path: Path):
    """Setup spatial mapping between graph nodes/edges and map regions.
    
    Launch interactive dual-view GUI for creating spatial mappings that link
    graph nodes and edges to regions on a map image.
    
    CONFIG_PATH: Path to configuration file containing graph and map settings
    
    \b
    The GUI provides two mapping modes:
      Grid Setup    - Place a grid and assign nodes/edges to cells
      Manual Drawing - Draw custom contours for each element
    
    \b
    Example:
      navigraph setup graph config.yaml
    
    \b
    Required config sections:
      setup.map_path: Path to the map image
      graph.builder: Defines graph type (binary_tree or custom)
      graph.mapping_file: Where to save the mapping
    """
    try:
        click.echo(f"Loading configuration: {config_path}")
        
        # Load configuration
        config = OmegaConf.load(config_path)
        config = process_config_path(config_path, OmegaConf.to_container(config))
        
        # Import graph modules
        from ..core.graph.structures import GraphStructure
        from ..core.graph.builders import get_graph_builder, list_graph_builders
        from ..core.graph.setup_gui_qt import launch_setup_gui
        from ..core.graph.storage import MappingStorage
        import numpy as np
        import cv2
        
        # Get map path from setup section
        setup_config = config.get('setup', {})
        map_path = setup_config.get('map_path')
        if not map_path:
            click.echo("Error: map_path not found in setup config section", err=True)
            click.echo("Please add 'map_path: path/to/map.png' under the 'setup:' section", err=True)
            sys.exit(1)
        
        # Resolve map path relative to config directory
        if not Path(map_path).is_absolute():
            map_path = Path(config['_config_dir']) / map_path
        
        # Load map image
        map_array = cv2.imread(str(map_path))
        if map_array is None:
            click.echo(f"Error: Failed to load map image: {map_path}", err=True)
            sys.exit(1)
        
        # Create graph from config using new builder system
        # Support both old 'graph_structure' and new 'graph.builder' formats
        graph_config = config.get('graph_structure', {})
        if not graph_config:
            # Try new format: graph.builder
            graph_section = config.get('graph', {})
            builder_config = graph_section.get('builder', {})
            if builder_config:
                graph_config = {
                    'type': builder_config.get('type'),
                    'parameters': builder_config.get('config', {})
                }
        
        graph_type = graph_config.get('type')
        if not graph_type:
            click.echo("Error: No graph builder type specified in configuration", err=True)
            click.echo("Add 'graph.builder.type' to your config file", err=True)
            click.echo(f"Available builders: {', '.join(list_graph_builders())}", err=True)
            sys.exit(1)
        
        # Get parameters for the builder
        params = graph_config.get('parameters', {})
        
        try:
            # Get builder class and create instance
            builder_class = get_graph_builder(graph_type)
            builder = builder_class(**params)
            
            # Create graph structure
            graph = GraphStructure(builder)
            
            # Get builder metadata for display
            metadata = builder.get_metadata()
            param_str = ', '.join(f"{k}={v}" for k, v in metadata['parameters'].items())
            
            # Display graph information
            click.echo(f"📊 Graph Builder: {graph_type}")
            click.echo(f"   Parameters: {param_str}")
            click.echo(f"   Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
            
        except KeyError as e:
            click.echo(f"Error: Unknown graph builder type '{graph_type}'", err=True)
            click.echo(f"Available builders: {', '.join(list_graph_builders())}", err=True)
            sys.exit(1)
        except TypeError as e:
            click.echo(f"Error: Invalid parameters for {graph_type} builder: {str(e)}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error: Failed to create graph: {str(e)}", err=True)
            sys.exit(1)
        
        # Launch PyQt5 GUI
        click.echo("Launching mapping interface...")
        
        try:
            mapping = launch_setup_gui(graph, map_array)
            
        except ImportError as e:
            click.echo(f"Error: PyQt5 is required but not installed: {e}", err=True)
            click.echo("Install with: pip install PyQt5", err=True)
            sys.exit(1)
            
        except Exception as e:
            click.echo(f"Error: Failed to launch interface: {e}", err=True)
            sys.exit(1)
        
        # GUI closed
        click.echo("Closing GUI...")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if '--verbose' in sys.argv:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


@setup.command('calibration')
@click.argument('config_path', type=click.Path(exists=True, path_type=Path))
@click.option("--test", is_flag=True, help="Test existing calibration instead of creating new one")
@click.option("--batch", is_flag=True, help="Calibrate each session individually")
@click.option("--shared", is_flag=True, help="Calibrate once using the first session, apply to all")
@click.option("--overwrite", is_flag=True, help="Overwrite existing calibrations in batch/shared mode")
def setup_calibration(config_path: Path, test: bool, batch: bool, shared: bool, overwrite: bool):
    """Setup camera calibration for spatial coordinate transformation.

    Launch interactive calibration tool to establish correspondence between
    camera view and map coordinates. This needs to be done whenever the
    camera position or angle changes.

    CONFIG_PATH: Path to configuration file

    \b
    Modes:
      (default)   Single session calibration using setup.spatial_image_for_calibration
      --batch     Calibrate each discovered session individually
      --shared    Calibrate once using the first session's video, copy to all sessions

    \b
    Example:
      navigraph setup calibration config.yaml
      navigraph setup calibration config.yaml --batch
      navigraph setup calibration config.yaml --shared --overwrite

    \b
    Required config sections:
      map_path: Path to the map image
      calibrator_parameters: Calibration settings
      experiment_path: (for --batch/--shared) Root directory containing session folders
    """
    if batch and shared:
        click.echo("Error: --batch and --shared are mutually exclusive.", err=True)
        sys.exit(1)

    try:
        click.echo(f"Loading configuration from: {config_path}")

        # Load configuration
        config = OmegaConf.load(config_path)
        config = process_config_path(config_path, OmegaConf.to_container(config))

        # Get map path from config (check setup section first, then root for backward compatibility)
        setup_config = config.get('setup', {})
        map_path = setup_config.get('map_path') or config.get('map_path')
        if not map_path:
            click.echo("Error: map_path not found in config. Add it to the setup section or root level.", err=True)
            sys.exit(1)

        # Resolve map path relative to config directory
        if not Path(map_path).is_absolute():
            map_path = Path(config['_config_dir']) / map_path

        click.echo(f"Map image: {map_path}")

        # Get calibration settings
        calib_params = config.get('calibrator_parameters', {})
        method = calib_params.get('registration_method', 'homography_ransac')

        # Convert legacy method name
        if method == 'homography&ransac':
            method = 'homography_ransac'

        min_points = calib_params.get('num_calibration_points', 4)

        if test:
            _run_calibration_test(config, setup_config, map_path)
        elif batch:
            _run_batch_calibration(config, map_path, method, min_points, overwrite, shared_mode=False)
        elif shared:
            _run_batch_calibration(config, map_path, method, min_points, overwrite, shared_mode=True)
        else:
            _run_single_calibration(config, setup_config, map_path, method, min_points)

    except Exception as e:
        click.echo(f"Calibration failed: {str(e)}", err=True)
        if '--verbose' in sys.argv:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)


def _run_calibration_test(config, setup_config, map_path):
    """Run calibration test mode."""
    click.echo("Testing existing calibration matrix")

    spatial_image_path = setup_config.get('spatial_image_for_calibration')
    if not spatial_image_path:
        click.echo("Error: spatial_image_for_calibration not found in setup section.", err=True)
        sys.exit(1)

    if not Path(spatial_image_path).is_absolute():
        spatial_image_path = Path(config['_config_dir']) / spatial_image_path

    # Get calibration matrix path from config
    calibration_matrix_path = setup_config.get('calibration_matrix')

    if not calibration_matrix_path:
        calibration_matrix_path = './resources/transform_matrix.npy'

    if not Path(calibration_matrix_path).is_absolute():
        calibration_matrix_path = Path(config['_config_dir']) / calibration_matrix_path

    if not Path(calibration_matrix_path).exists():
        click.echo(f"Calibration matrix not found: {calibration_matrix_path}", err=True)
        click.echo("Run calibration without --test to create one first", err=True)
        sys.exit(1)

    click.echo(f"Testing calibration: {calibration_matrix_path}")

    from ..core.calibration import CalibrationTester

    tester = CalibrationTester()
    tester.test_calibration(
        spatial_image_path=spatial_image_path,
        map_image_path=map_path,
        calibration_matrix_path=calibration_matrix_path
    )

    click.echo("Calibration test completed")


def _run_single_calibration(config, setup_config, map_path, method, min_points):
    """Run single session calibration (original behavior)."""
    spatial_image_path = setup_config.get('spatial_image_for_calibration')

    if not spatial_image_path:
        click.echo("Error: spatial_image_for_calibration not found in setup section.", err=True)
        click.echo("Add it to config: setup.spatial_image_for_calibration: /path/to/video/or/image", err=True)
        sys.exit(1)

    if not Path(spatial_image_path).is_absolute():
        spatial_image_path = Path(config['_config_dir']) / spatial_image_path

    click.echo(f"Spatial source: {spatial_image_path}")
    click.echo(f"Method: {method}")
    click.echo(f"Minimum points: {min_points}")

    from ..core.calibration import InteractiveCalibrator

    calibrator = InteractiveCalibrator()

    calibration_result = calibrator.calibrate_camera_to_map(
        camera_source=spatial_image_path,
        map_image_path=map_path,
        method=method,
        min_points=min_points,
        show_preview=True
    )

    # Save to resources directory
    output_dir = Path(config['_config_dir']) / 'resources'
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / 'transform_matrix.npy'
    calibration_result.save(matrix_path)

    click.echo("Calibration completed successfully!")


def _find_session_video(session_path: Path) -> 'Optional[Path]':
    """Find the first .avi video file in a session directory.

    Args:
        session_path: Path to the session directory

    Returns:
        Path to the first .avi file found, or None
    """
    avi_files = sorted(session_path.glob("*.avi"))
    if avi_files:
        return avi_files[0]
    return None


def _run_batch_calibration(config, map_path, method, min_points, overwrite, shared_mode):
    """Run batch or shared calibration across all sessions.

    Args:
        config: Loaded configuration dictionary
        map_path: Path to the map image
        method: Transform method string
        min_points: Minimum calibration points
        overwrite: Whether to overwrite existing calibrations
        shared_mode: If True, calibrate once and copy to all sessions.
                     If False, calibrate each session individually.
    """
    from ..core.file_discovery import FileDiscoveryEngine
    from ..core.calibration import InteractiveCalibrator
    import shutil

    experiment_path = config.get('experiment_path')
    if not experiment_path:
        click.echo("Error: experiment_path not found in config. Required for --batch/--shared mode.", err=True)
        sys.exit(1)

    experiment_path = Path(experiment_path)
    if not experiment_path.exists():
        click.echo(f"Error: experiment_path does not exist: {experiment_path}", err=True)
        sys.exit(1)

    # Discover sessions
    from loguru import logger as _logger
    discovery = FileDiscoveryEngine(str(experiment_path), _logger)
    session_names = discovery.discover_session_folders()

    if not session_names:
        click.echo("No sessions found in experiment directory.", err=True)
        sys.exit(1)

    click.echo(f"Found {len(session_names)} sessions")

    # Determine output directory name from config (default: navigraph_output)
    output_dir_name = config.get('output_directory_name', 'navigraph_output')

    mode_label = "shared" if shared_mode else "batch"
    click.echo(f"Running {mode_label} calibration across {len(session_names)} sessions")

    calibrator = InteractiveCalibrator()
    shared_result = None  # Stores the calibration result in shared mode

    completed = 0
    skipped = 0

    for i, session_name in enumerate(session_names, 1):
        session_path = experiment_path / session_name
        output_dir = session_path / output_dir_name
        matrix_path = output_dir / 'transform_matrix.npy'

        # Check if calibration already exists
        if matrix_path.exists() and not overwrite:
            click.echo(f"[{i}/{len(session_names)}] {session_name} -- skipped (already calibrated)")
            skipped += 1
            continue

        if shared_mode and shared_result is not None:
            # In shared mode, copy the already-computed calibration
            output_dir.mkdir(parents=True, exist_ok=True)
            shared_result.save(matrix_path)
            click.echo(f"[{i}/{len(session_names)}] {session_name} -- copied shared calibration")
            completed += 1
            continue

        # Find video file for this session
        video_path = _find_session_video(session_path)
        if video_path is None:
            click.echo(f"[{i}/{len(session_names)}] {session_name} -- skipped (no .avi file found)")
            skipped += 1
            continue

        click.echo(f"[{i}/{len(session_names)}] {session_name} -- calibrating with {video_path.name}")

        try:
            calibration_result = calibrator.calibrate_camera_to_map(
                camera_source=video_path,
                map_image_path=map_path,
                method=method,
                min_points=min_points,
                show_preview=True
            )

            output_dir.mkdir(parents=True, exist_ok=True)
            calibration_result.save(matrix_path)
            click.echo(f"  Saved: {matrix_path}")
            completed += 1

            if shared_mode:
                shared_result = calibration_result
                click.echo(f"  (shared mode: this calibration will be applied to remaining sessions)")

        except ValueError as e:
            if "cancelled" in str(e).lower():
                click.echo(f"  Skipped (cancelled by user)")
                skipped += 1
                continue
            else:
                raise

    click.echo(f"\n{mode_label.capitalize()} calibration complete: {completed} calibrated, {skipped} skipped")




@cli.command('list-conflict-resolvers')
@click.option('--format', '-f', 
              type=click.Choice(['table', 'json', 'simple']),
              default='table',
              help='Output format')
def list_conflict_resolvers(format: str):
    """List all available conflict resolution strategies.
    
    Display registered conflict resolvers that can be used in spatial mapping
    when pixels fall within multiple regions.
    
    \b
    Examples:
      navigraph list-conflict-resolvers
      navigraph list-conflict-resolvers --format json
    """
    try:
        from ..core.graph.conflict_resolvers import ConflictResolvers
        
        strategies = ConflictResolvers._strategies
        
        if not strategies:
            click.echo("⚠️  No conflict resolvers found")
            return
        
        if format == 'simple':
            for name in strategies.keys():
                click.echo(name)
        
        elif format == 'json':
            output = {
                'available_resolvers': [
                    {
                        'name': name,
                        'description': func.__doc__.strip() if func.__doc__ else ''
                    }
                    for name, func in strategies.items()
                ],
                'total_count': len(strategies),
                'default': 'node_priority'
            }
            click.echo(json.dumps(output, indent=2))
        
        else:  # table format
            click.echo()
            click.echo("🔀 Available Conflict Resolution Strategies")
            click.echo("=" * 55)
            
            for name, func in strategies.items():
                click.echo(f"\n🔹 {name}")
                if func.__doc__:
                    doc = func.__doc__.strip().split('\n')[0]
                    click.echo(f"   {doc}")
            
            click.echo()
            click.echo(f"Total: {len(strategies)} strategies available")
            click.echo(f"Default: node_priority")
            click.echo()
            click.echo("Usage in config:")
            click.echo("  shared_resources:")
            click.echo("    - name: graph_provider")
            click.echo("      config:")
            click.echo("        conflict_strategy: <resolver_name>")
            click.echo()
            click.echo("CLI override:")
            click.echo("  navigraph run config.yaml --conflict-strategy <resolver_name>")
            
    except Exception as e:
        click.echo(f"❌ Failed to list conflict resolvers: {str(e)}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the NaviGraph CLI."""
    cli()


if __name__ == '__main__':
    main()