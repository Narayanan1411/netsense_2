import logging
import json
import os
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from numba import njit

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("network_optimizer_enhanced")

# Default configuration constants
DEFAULT_MAX_DRIVE_MIN = 30.0
DEFAULT_MIN_COVERAGE_PCT = 95.0
DEFAULT_MIN_AVG_RATING = 3.0
TARGET_REDUCTION = (0.08, 0.12)
MINUTES_PER_KM = 1.2
K_NEAREST_PROVIDERS = 50
MAX_REMOVAL_CANDIDATES = 5000

# Provider type mappings
TYPE_CANONICAL = {
    "hospital": ["hospital", "hosp", "acute care", "psychiatric", "rehabilitation", "children"],
    "nursing_home": ["nursing", "home health", "assisted living", "foster care", "hospice", "residential care", "skilled nursing"],
    "scan_center": ["scan", "imaging", "mri", "ct", "x-ray", "ultrasound"],
    "clinic": ["clinic", "outpatient", "primary care", "pediatrician", "optometrist", "obstetric"],
    "supplier_directory": ["supplier", "directory", "pharmacy", "medical supply", "optical", "durable medical", "optician", "orthotic", "prosthetic"],
    "other": ["grocery", "department store"]
}

TYPE_MIN_COUNTS = {
    "hospital": 1,
    "nursing_home": 0,
    "scan_center": 0,
    "clinic": 1,
    "supplier_directory": 0
}

@njit
def haversine_km_numba(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in kilometers using Numba JIT"""
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Find column name from candidates"""
    cols = list(df.columns)
    lowered = {c.lower().replace(" ", "").replace("_", ""): c for c in cols}
    
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in lowered:
            return lowered[key]
    
    for cand in candidates:
        if cand in cols:
            return cand
    
    return None

def map_columns_providers(df: pd.DataFrame) -> pd.DataFrame:
    """Map provider columns to standard names"""
    col_map = {
        "ProviderId": find_col(df, ["providerid", "provider id", "provider_id", "id"]),
        "Source": find_col(df, ["source"]),
        "Type": find_col(df, ["type", "facilitytype", "facility_type"]),
        "Latitude": find_col(df, ["latitude", "lat", "latt"]),
        "Longitude": find_col(df, ["longitude", "lon", "lng"]),
        "CMS_Rating": find_col(df, ["cmsrating", "cms rating", "rating", "star", "cms_rating"]),
        "Availability": find_col(df, ["availability", "capacity", "avail"]),
        "Cost": find_col(df, ["cost", "annualcost", "contractcost", "annual_cost", "contract_cost"])
    }
    
    rename_map = {v: k for k, v in col_map.items() if v}
    mapped_df = df.rename(columns=rename_map)
    
    # Validate required columns
    required_cols = ["ProviderId", "Latitude", "Longitude"]
    missing_cols = [col for col in required_cols if col not in mapped_df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required provider columns: {missing_cols}")
    
    # Fill missing optional columns with defaults
    if "CMS_Rating" not in mapped_df.columns:
        mapped_df["CMS_Rating"] = 3.0
    if "Cost" not in mapped_df.columns:
        mapped_df["Cost"] = 100000.0
    if "Availability" not in mapped_df.columns:
        mapped_df["Availability"] = 1.0
    
    return mapped_df

def map_columns_members(df: pd.DataFrame) -> pd.DataFrame:
    """Map member columns to standard names"""
    col_map = {
        "MemberId": find_col(df, ["memberid", "member id", "member_id", "id"]),
        "Latitude": find_col(df, ["latitude", "lat"]),
        "Longitude": find_col(df, ["longitude", "lon", "lng"])
    }
    
    rename_map = {v: k for k, v in col_map.items() if v}
    mapped_df = df.rename(columns=rename_map)
    
    # Validate required columns
    required_cols = ["MemberId", "Latitude", "Longitude"]
    missing_cols = [col for col in required_cols if col not in mapped_df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required member columns: {missing_cols}")
    
    return mapped_df

def normalize_provider_type(t: Any) -> str:
    """Normalize provider type to canonical form"""
    if pd.isna(t):
        return "other"
    
    s = str(t).strip().lower()
    for canon, examples in TYPE_CANONICAL.items():
        for ex in examples:
            if ex in s:
                return canon
    
    return "other"

def apply_type_normalization(providers: pd.DataFrame) -> pd.DataFrame:
    """Apply provider type normalization"""
    prov = providers.copy()
    raw = prov.get("Type", prov.get("Source", pd.Series(["other"] * len(prov))))
    prov["ProviderType"] = raw.fillna("other").astype(str).apply(normalize_provider_type)
    return prov

def build_candidate_pairs(members: pd.DataFrame, providers: pd.DataFrame, max_drive_min: float) -> pd.DataFrame:
    """Build member-provider candidate pairs using k-d tree"""
    if members.empty or providers.empty:
        return pd.DataFrame(columns=['MemberId', 'ProviderId', 'DriveTimeMin'])
    
    logger.info("Building provider k-d tree for fast spatial queries...")
    provider_coords = providers[['Latitude', 'Longitude']].to_numpy()
    provider_tree = cKDTree(provider_coords)
    
    member_coords = members[['Latitude', 'Longitude']].to_numpy()
    max_dist_in_degrees = max_drive_min / MINUTES_PER_KM / 111.0
    
    logger.info(f"Querying for the {K_NEAREST_PROVIDERS} nearest providers for each member...")
    distances, indices = provider_tree.query(
        member_coords, 
        k=K_NEAREST_PROVIDERS, 
        distance_upper_bound=max_dist_in_degrees, 
        workers=-1
    )
    
    logger.info("Filtering and building final candidate list...")
    valid_pairs = []
    member_ids = members['MemberId'].values
    provider_ids = providers['ProviderId'].values
    
    for i in range(len(member_coords)):
        for j in range(K_NEAREST_PROVIDERS):
            provider_idx = indices[i, j]
            if provider_idx < len(providers):
                dist_km = haversine_km_numba(
                    member_coords[i, 0], member_coords[i, 1],
                    provider_coords[provider_idx, 0], provider_coords[provider_idx, 1]
                )
                drive_time = dist_km * MINUTES_PER_KM
                if drive_time <= max_drive_min:
                    valid_pairs.append((member_ids[i], provider_ids[provider_idx], drive_time))
    
    if not valid_pairs:
        logger.warning("No member-provider pairs found within %.1f minutes.", max_drive_min)
        return pd.DataFrame(columns=['MemberId', 'ProviderId', 'DriveTimeMin'])
    
    return pd.DataFrame(valid_pairs, columns=['MemberId', 'ProviderId', 'DriveTimeMin'])

def interpret_capacity(providers: pd.DataFrame, members_count: int) -> pd.DataFrame:
    """Interpret provider availability as capacity"""
    prov = providers.copy()
    n_prov = max(1, len(providers))
    avg_panel_size = max(1, int(round(members_count / n_prov)))
    
    def calc_cap(avail):
        if avail <= 1.0:
            return max(1, int(round(avail * avg_panel_size)))
        return max(1, int(round(avail)))
    
    prov["Capacity"] = prov["Availability"].astype(float).fillna(0.0).apply(calc_cap)
    return prov

def assign_greedy(members: pd.DataFrame, providers: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Greedy assignment algorithm"""
    if candidates.empty:
        return pd.DataFrame(columns=["MemberId", "ProviderId"])
    
    # Sort candidates by preference
    cand = candidates.sort_values(
        ["MemberId", "DriveTimeMin", "CMS_Rating", "Cost"], 
        ascending=[True, True, False, True]
    )
    
    capacity_map = providers.set_index("ProviderId")["Capacity"].to_dict()
    assignments = []
    assigned_members = set()
    
    for _, row in cand.iterrows():
        member_id = row['MemberId']
        provider_id = row['ProviderId']
        
        if member_id not in assigned_members and capacity_map.get(provider_id, 0) > 0:
            assignments.append((member_id, provider_id))
            capacity_map[provider_id] -= 1
            assigned_members.add(member_id)
    
    return pd.DataFrame(assignments, columns=["MemberId", "ProviderId"])

def evaluate(assignments: pd.DataFrame, members: pd.DataFrame, providers: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate assignment quality"""
    total_members = len(members)
    
    if assignments.empty:
        return {
            "coverage_pct": 0.0,
            "avg_rating": 0.0,
            "total_cost": 0.0,
            "providers_used": 0,
            "final_assignments": pd.DataFrame()
        }
    
    covered_members = assignments["MemberId"].nunique()
    coverage_pct = (covered_members / total_members * 100.0) if total_members else 0.0
    
    used_provider_counts = assignments["ProviderId"].value_counts()
    used_providers = providers[providers["ProviderId"].isin(used_provider_counts.index)].copy()
    
    avg_rating = used_providers["CMS_Rating"].mean() if not used_providers.empty else 0.0
    total_cost = used_providers["Cost"].sum() if not used_providers.empty else 0.0
    
    final_assignments = assignments.merge(
        providers[["ProviderId", "ProviderType", "CMS_Rating", "Cost"]], 
        on="ProviderId"
    )
    
    return {
        "coverage_pct": float(coverage_pct),
        "avg_rating": float(avg_rating),
        "total_cost": float(total_cost),
        "providers_used": len(used_providers),
        "final_assignments": final_assignments
    }

def compute_removal_priority(assignments: pd.DataFrame, candidates: pd.DataFrame, providers: pd.DataFrame) -> pd.DataFrame:
    """Compute provider removal priority scores"""
    provider_assignments = assignments.groupby("ProviderId")["MemberId"].apply(list).to_dict()
    member_alternatives = candidates.groupby("MemberId")["ProviderId"].nunique().to_dict()
    
    rows = []
    for _, p in providers.iterrows():
        assigned_mids = provider_assignments.get(p.ProviderId, [])
        num_assigned = len(assigned_mids)
        num_unique = sum(1 for mid in assigned_mids if member_alternatives.get(mid, 1) <= 1)
        
        cost_per_assigned = p.Cost / max(1, num_assigned)
        score = cost_per_assigned - (p.CMS_Rating * 1000) + (num_unique * 10000)
        
        rows.append((p.ProviderId, score))
    
    return pd.DataFrame(rows, columns=["ProviderId", "Score"]).sort_values("Score", ascending=False)

class NetworkOptimizer:
    """Provider Network Optimizer with intelligent reassignment algorithm"""
    
    def __init__(self, max_drive_min=DEFAULT_MAX_DRIVE_MIN, min_coverage_pct=DEFAULT_MIN_COVERAGE_PCT, min_avg_rating=DEFAULT_MIN_AVG_RATING):
        self.max_drive_min = max_drive_min
        self.min_coverage_pct = min_coverage_pct
        self.min_avg_rating = min_avg_rating
        self.progress_handler = None
        
    def optimize(self, members: pd.DataFrame, providers: pd.DataFrame) -> Dict[str, Any]:
        """Main optimization function"""
        try:
            logger.info("Starting network optimization...")
            
            # Map and validate columns
            providers_mapped = map_columns_providers(providers)
            members_mapped = map_columns_members(members)
            
            # Run optimization
            result = self._safe_prune(members_mapped, providers_mapped)
            
            logger.info("Network optimization completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise
    
    def _safe_prune(self, members: pd.DataFrame, providers: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced optimization algorithm with incremental reassignment"""
        providers_with_capacity = interpret_capacity(providers, len(members))
        providers_with_types = apply_type_normalization(providers_with_capacity)
        
        # Pre-calculate all possible member-provider options
        candidates = build_candidate_pairs(members, providers_with_types, self.max_drive_min)
        if candidates.empty:
            raise RuntimeError(f"No viable member-provider pairs found within {self.max_drive_min} minutes.")
        
        # Merge provider data into candidates for fast lookups
        candidates_with_attrs = candidates.merge(
            providers_with_types[["ProviderId", "CMS_Rating", "Cost", "Capacity", "ProviderType"]], 
            on="ProviderId"
        )
        
        # Establish baseline
        base_assign = assign_greedy(members, providers_with_types, candidates_with_attrs)
        base_kpi = evaluate(base_assign, members, providers_with_types)
        baseline_cost = base_kpi["total_cost"]
        
        logger.info(f"Baseline: coverage={base_kpi['coverage_pct']:.2f}% | avg_rating={base_kpi['avg_rating']:.2f} | cost=${baseline_cost:,.2f} | providers={base_kpi['providers_used']}")
        
        if base_kpi["coverage_pct"] < self.min_coverage_pct:
            raise RuntimeError(f"Baseline coverage is {base_kpi['coverage_pct']:.2f}%, below required {self.min_coverage_pct}%.")
        
        removal_candidates = compute_removal_priority(base_assign, candidates, providers_with_types).head(MAX_REMOVAL_CANDIDATES)
        logger.info(f"Considering the top {MAX_REMOVAL_CANDIDATES} providers for removal...")
        
        current_assignments = base_assign.copy()
        current_providers_set = set(providers_with_types['ProviderId'])
        removed_pids = []
        
        provider_info = providers_with_types.set_index('ProviderId')
        
        # Main optimization loop
        for i, (_, row) in enumerate(removal_candidates.iterrows()):
            if i % 50 == 0 and i > 0:
                logger.info(f"Processing removal candidate {i}/{len(removal_candidates)}...")
            
            pid_to_remove = row.ProviderId
            
            # Fast check for type constraint violation
            provider_type = provider_info.loc[pid_to_remove, 'ProviderType']
            type_count = provider_info.loc[list(current_providers_set)].ProviderType.value_counts().get(provider_type, 0)
            
            if type_count - 1 < TYPE_MIN_COUNTS.get(provider_type, 0):
                continue
            
            # Identify members who need reassignment
            members_to_reassign = current_assignments[current_assignments['ProviderId'] == pid_to_remove]
            
            if members_to_reassign.empty:
                # If provider had no assignments, we can safely remove them if cost is positive
                if provider_info.loc[pid_to_remove, 'Cost'] > 0:
                    current_providers_set.remove(pid_to_remove)
                    removed_pids.append(pid_to_remove)
                continue
            
            # Temporarily remove the provider and update assignments
            trial_providers_set = current_providers_set - {pid_to_remove}
            other_assignments = current_assignments[current_assignments['ProviderId'] != pid_to_remove]
            
            # Find new assignments ONLY for the displaced members
            reassign_candidates = candidates_with_attrs[
                candidates_with_attrs['MemberId'].isin(members_to_reassign['MemberId']) &
                candidates_with_attrs['ProviderId'].isin(trial_providers_set)
            ]
            
            # Fast, small-scale greedy assignment
            new_assignments = assign_greedy(
                members_to_reassign, 
                provider_info.loc[list(trial_providers_set)].reset_index(), 
                reassign_candidates
            )
            
            # Combine to form the new trial assignment table
            trial_assignments = pd.concat([other_assignments, new_assignments], ignore_index=True)
            
            # Evaluate trial
            trial_kpi = evaluate(trial_assignments, members, provider_info.loc[list(trial_providers_set)].reset_index())
            
            # Accept if constraints are met
            if (trial_kpi["coverage_pct"] >= self.min_coverage_pct and 
                trial_kpi["avg_rating"] >= self.min_avg_rating):
                
                current_assignments = trial_assignments
                current_providers_set = trial_providers_set
                removed_pids.append(pid_to_remove)
                
                # Log progress and update Streamlit if handler available
                message = f"Removed provider {pid_to_remove}. New cost: ${trial_kpi['total_cost']:,.2f}"
                logger.info(message)
                if self.progress_handler:
                    self.progress_handler.update_progress(message)
        
        # Final evaluation
        final_providers = provider_info.loc[list(current_providers_set)].reset_index()
        final_kpi = evaluate(current_assignments, members, final_providers)
        
        savings = baseline_cost - final_kpi["total_cost"]
        savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0
        
        logger.info(f"Optimization complete. Removed {len(removed_pids)} providers.")
        logger.info(f"Final: coverage={final_kpi['coverage_pct']:.2f}% | avg_rating={final_kpi['avg_rating']:.2f} | cost=${final_kpi['total_cost']:,.2f} | savings=${savings:,.2f} ({savings_pct:.1f}%)")
        
        return final_kpi