"""
Task 1: Domain ontology definition.

An ontology specifies *classes* (types of things) and *relationship types*
(how those things connect). This file is the single source of truth for labels
used during knowledge-graph construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EntityClass:
    name: str
    description: str
    examples: tuple[str, ...]
    attributes: tuple[str, ...]


@dataclass(frozen=True)
class RelationshipType:
    name: str
    description: str
    domain: str
    range: str
    example: str


@dataclass
class DomainOntology:
    domain: str
    entity_classes: dict[str, EntityClass] = field(default_factory=dict)
    relationship_types: dict[str, RelationshipType] = field(default_factory=dict)

    def summary_table(self) -> list[dict]:
        rows = []
        for cls in self.entity_classes.values():
            rows.append(
                {
                    "Type": "Entity Class",
                    "Name": cls.name,
                    "Description": cls.description,
                    "Attributes / Examples": ", ".join(cls.attributes + cls.examples),
                }
            )
        for rel in self.relationship_types.values():
            rows.append(
                {
                    "Type": "Relationship",
                    "Name": rel.name,
                    "Description": rel.description,
                    "Attributes / Examples": f"{rel.domain} -> {rel.range}; e.g. {rel.example}",
                }
            )
        return rows


def build_climate_ontology() -> DomainOntology:
    """Return the climate-domain ontology used across tasks."""
    entity_classes = {
        "ClimatePhenomenon": EntityClass(
            name="ClimatePhenomenon",
            description="Observable climate processes or impacts such as warming or sea-level rise.",
            examples=("global warming", "sea level rise", "ocean acidification"),
            attributes=("severity", "time_scale", "geographic_scope"),
        ),
        "GreenhouseGas": EntityClass(
            name="GreenhouseGas",
            description="Atmospheric gases that trap heat and contribute to radiative forcing.",
            examples=("carbon dioxide", "methane", "nitrous oxide"),
            attributes=("chemical_formula", "global_warming_potential", "lifetime"),
        ),
        "PolicyInstrument": EntityClass(
            name="PolicyInstrument",
            description="Legal or economic mechanisms designed to mitigate or adapt to climate change.",
            examples=("Paris Agreement", "carbon tax", "emissions trading"),
            attributes=("jurisdiction", "start_year", "target"),
        ),
        "Organization": EntityClass(
            name="Organization",
            description="Institutions that research, govern, or coordinate climate action.",
            examples=("IPCC", "UNFCCC", "World Meteorological Organization"),
            attributes=("role", "founding_year", "headquarters"),
        ),
        "GeographicRegion": EntityClass(
            name="GeographicRegion",
            description="Places or ecosystems affected by or relevant to climate processes.",
            examples=("Arctic", "Amazon rainforest", "Indian Ocean"),
            attributes=("latitude_band", "biome", "population"),
        ),
        "EnergySource": EntityClass(
            name="EnergySource",
            description="Technologies or fuels used for energy production with distinct emissions profiles.",
            examples=("solar power", "wind power", "coal"),
            attributes=("carbon_intensity", "capacity_factor", "cost"),
        ),
    }

    relationship_types = {
        "causes": RelationshipType(
            name="causes",
            description="One entity directly contributes to another climate process or impact.",
            domain="GreenhouseGas | EnergySource",
            range="ClimatePhenomenon",
            example="(carbon dioxide, causes, global warming)",
        ),
        "mitigated_by": RelationshipType(
            name="mitigated_by",
            description="A phenomenon or emission source is reduced through a policy or technology.",
            domain="ClimatePhenomenon | GreenhouseGas",
            range="PolicyInstrument | EnergySource",
            example="(deforestation, mitigated_by, reforestation policy)",
        ),
        "regulated_by": RelationshipType(
            name="regulated_by",
            description="An activity or emission is governed by an institution or treaty.",
            domain="GreenhouseGas | EnergySource",
            range="PolicyInstrument | Organization",
            example="(methane emissions, regulated_by, Paris Agreement)",
        ),
        "located_in": RelationshipType(
            name="located_in",
            description="A phenomenon, organization activity, or resource is associated with a region.",
            domain="ClimatePhenomenon | Organization",
            range="GeographicRegion",
            example="(Arctic sea ice decline, located_in, Arctic)",
        ),
        "emits": RelationshipType(
            name="emits",
            description="An energy source or sector releases greenhouse gases.",
            domain="EnergySource",
            range="GreenhouseGas",
            example="(coal, emits, carbon dioxide)",
        ),
        "contributes_to": RelationshipType(
            name="contributes_to",
            description="An organization or report synthesizes evidence about a phenomenon.",
            domain="Organization | PolicyInstrument",
            range="ClimatePhenomenon",
            example="(IPCC, contributes_to, attribution of recent climate change)",
        ),
    }

    return DomainOntology(
        domain="Climate Change and Environmental Science",
        entity_classes=entity_classes,
        relationship_types=relationship_types,
    )
