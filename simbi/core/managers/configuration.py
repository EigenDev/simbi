from dataclasses import dataclass
from typing import Any
from ...functional.maybe import Maybe
from ..config.base_config import BaseConfig
from .cli import CLIManager
from .property import PropertyManager
from .source import SourceManager


@dataclass
class ConfigurationManager:
    """Coordinates all configuration managers"""

    cli: CLIManager
    property: PropertyManager
    source: SourceManager

    def _process_cli_args(self, config: BaseConfig) -> Maybe[BaseConfig]:
        """Process command line arguments"""
        return Maybe.of(self.cli.parse_args()).map(config.update)
    
    def _compile_sources(self, config: BaseConfig) -> Maybe[BaseConfig]:
        """Compile source code"""
        if not config.sources:
            return Maybe.of(config)
        
        compiled_sources = self.source.compile_sources(
            config.class_name, config.sources
        )
        return Maybe.of(config.update(compiled_sources=compiled_sources))
    
    def process_config(self, config: "BaseConfig") -> Maybe[dict[str, Any]]:
        """Process and validate configuration"""
        return (
            Maybe.of(config)
            .bind(self._process_cli_args)
            .bind(self._compile_sources)
            .bind(lambda c: c.to_settings())
        )
