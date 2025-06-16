import json
import os
from datetime import datetime
from typing import List, Dict, Optional

class AuditLogger:
    """Manages audit trail logging for all application activities"""
    
    def __init__(self):
        self.data_dir = "data"
        self.audit_file = os.path.join(self.data_dir, "audit_log.json")
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize audit log if it doesn't exist
        if not os.path.exists(self.audit_file):
            self._initialize_audit_log()
    
    def _initialize_audit_log(self):
        """Initialize empty audit log file"""
        initial_log = {
            "created": datetime.now().isoformat(),
            "version": "1.0",
            "entries": []
        }
        
        try:
            with open(self.audit_file, 'w') as f:
                json.dump(initial_log, f, indent=2)
        except Exception as e:
            print(f"Failed to initialize audit log: {str(e)}")
    
    def _add_log_entry(self, action: str, details: Dict, metadata: Optional[Dict] = None):
        """Add a new entry to the audit log"""
        try:
            # Load existing log
            with open(self.audit_file, 'r') as f:
                audit_log = json.load(f)
            
            # Create new entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "details": details,
                "metadata": metadata or {}
            }
            
            # Add to entries
            audit_log["entries"].append(entry)
            
            # Keep only last 1000 entries to prevent file from growing too large
            if len(audit_log["entries"]) > 1000:
                audit_log["entries"] = audit_log["entries"][-1000:]
            
            # Save back to file
            with open(self.audit_file, 'w') as f:
                json.dump(audit_log, f, indent=2)
                
        except Exception as e:
            print(f"Failed to add audit log entry: {str(e)}")
    
    def log_upload(self, filename: str, transaction_count: int):
        """Log PDF upload and processing"""
        self._add_log_entry(
            action="PDF_UPLOAD",
            details={
                "filename": filename,
                "transaction_count": transaction_count,
                "status": "success"
            },
            metadata={
                "file_size": "unknown",  # Could be enhanced to include file size
                "processing_time": "unknown"  # Could be enhanced to include processing time
            }
        )
    
    def log_categorization_change(self, narration: str, old_category: str, new_category: str):
        """Log manual categorization changes"""
        self._add_log_entry(
            action="CATEGORIZATION_CHANGE",
            details={
                "narration": narration[:100],  # Truncate long narrations
                "old_category": old_category,
                "new_category": new_category,
                "change_type": "manual_correction"
            }
        )
    
    def log_category_creation(self, category_name: str, keywords: List[str]):
        """Log creation of custom categories"""
        self._add_log_entry(
            action="CATEGORY_CREATION",
            details={
                "category_name": category_name,
                "keywords": keywords,
                "keyword_count": len(keywords)
            }
        )
    
    def log_category_deletion(self, category_name: str):
        """Log deletion of custom categories"""
        self._add_log_entry(
            action="CATEGORY_DELETION",
            details={
                "category_name": category_name
            }
        )
    
    def log_data_export(self, export_type: str, record_count: int):
        """Log data export activities"""
        self._add_log_entry(
            action="DATA_EXPORT",
            details={
                "export_type": export_type,  # "transactions", "audit_log", etc.
                "record_count": record_count,
                "format": "CSV"
            }
        )
    
    def log_cache_operation(self, operation: str, details: Dict):
        """Log AI cache operations"""
        self._add_log_entry(
            action="CACHE_OPERATION",
            details={
                "operation": operation,  # "clear", "import", "export"
                **details
            }
        )
    
    def log_suspicious_detection(self, transaction_count: int, detection_rules: List[str]):
        """Log suspicious transaction detection"""
        self._add_log_entry(
            action="SUSPICIOUS_DETECTION",
            details={
                "suspicious_count": transaction_count,
                "detection_rules": detection_rules,
                "status": "completed"
            }
        )
    
    def log_data_clear(self, data_type: str):
        """Log data clearing operations"""
        self._add_log_entry(
            action="DATA_CLEAR",
            details={
                "data_type": data_type,  # "transactions", "all_data", "cache"
                "status": "completed"
            }
        )
    
    def log_application_start(self):
        """Log application startup"""
        self._add_log_entry(
            action="APPLICATION_START",
            details={
                "status": "started"
            },
            metadata={
                "version": "1.0"
            }
        )
    
    def log_error(self, error_type: str, error_message: str, context: Dict):
        """Log application errors"""
        self._add_log_entry(
            action="ERROR",
            details={
                "error_type": error_type,
                "error_message": error_message[:200],  # Truncate long error messages
                "context": context
            }
        )
    
    def get_audit_log(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve audit log entries
        
        Args:
            limit: Maximum number of entries to return (most recent first)
            
        Returns:
            List of audit log entries
        """
        try:
            with open(self.audit_file, 'r') as f:
                audit_log = json.load(f)
            
            entries = audit_log.get("entries", [])
            
            # Sort by timestamp (most recent first)
            entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            if limit:
                entries = entries[:limit]
            
            return entries
            
        except Exception as e:
            print(f"Failed to retrieve audit log: {str(e)}")
            return []
    
    def get_audit_summary(self) -> Dict:
        """Get summary statistics of audit log"""
        try:
            entries = self.get_audit_log()
            
            if not entries:
                return {
                    "total_entries": 0,
                    "date_range": {"start": None, "end": None},
                    "action_counts": {},
                    "recent_activity": []
                }
            
            # Count actions
            action_counts = {}
            for entry in entries:
                action = entry.get("action", "unknown")
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Get date range
            timestamps = [entry.get("timestamp") for entry in entries if entry.get("timestamp")]
            timestamps.sort()
            
            return {
                "total_entries": len(entries),
                "date_range": {
                    "start": timestamps[0] if timestamps else None,
                    "end": timestamps[-1] if timestamps else None
                },
                "action_counts": action_counts,
                "recent_activity": entries[:10]  # Last 10 activities
            }
            
        except Exception as e:
            print(f"Failed to get audit summary: {str(e)}")
            return {
                "total_entries": 0,
                "date_range": {"start": None, "end": None},
                "action_counts": {},
                "recent_activity": []
            }
    
    def search_audit_log(self, 
                        action_filter: Optional[str] = None,
                        date_from: Optional[str] = None,
                        date_to: Optional[str] = None,
                        keyword: Optional[str] = None) -> List[Dict]:
        """
        Search audit log with filters
        
        Args:
            action_filter: Filter by action type
            date_from: Filter from date (ISO format)
            date_to: Filter to date (ISO format)
            keyword: Search keyword in details
            
        Returns:
            Filtered list of audit entries
        """
        try:
            entries = self.get_audit_log()
            filtered_entries = []
            
            for entry in entries:
                # Action filter
                if action_filter and entry.get("action") != action_filter:
                    continue
                
                # Date range filter
                entry_timestamp = entry.get("timestamp")
                if date_from and entry_timestamp < date_from:
                    continue
                if date_to and entry_timestamp > date_to:
                    continue
                
                # Keyword search
                if keyword:
                    entry_text = json.dumps(entry).lower()
                    if keyword.lower() not in entry_text:
                        continue
                
                filtered_entries.append(entry)
            
            return filtered_entries
            
        except Exception as e:
            print(f"Failed to search audit log: {str(e)}")
            return []
    
    def export_audit_log(self) -> str:
        """Export audit log as JSON string"""
        try:
            with open(self.audit_file, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Failed to export audit log: {str(e)}")
            return json.dumps({"error": "Failed to export audit log"})
    
    def clear_log(self):
        """Clear the audit log (keep structure)"""
        try:
            cleared_log = {
                "created": datetime.now().isoformat(),
                "version": "1.0",
                "entries": [],
                "cleared_at": datetime.now().isoformat()
            }
            
            with open(self.audit_file, 'w') as f:
                json.dump(cleared_log, f, indent=2)
                
        except Exception as e:
            print(f"Failed to clear audit log: {str(e)}")
    
    def get_compliance_report(self) -> Dict:
        """Generate a compliance report from audit log"""
        try:
            entries = self.get_audit_log()
            summary = self.get_audit_summary()
            
            # Count critical operations
            critical_operations = ["DATA_CLEAR", "CACHE_OPERATION", "CATEGORY_DELETION"]
            critical_count = sum(
                summary["action_counts"].get(action, 0) 
                for action in critical_operations
            )
            
            # Get file operations
            file_operations = [
                entry for entry in entries 
                if entry.get("action") in ["PDF_UPLOAD", "DATA_EXPORT"]
            ]
            
            return {
                "report_generated": datetime.now().isoformat(),
                "total_activities": summary["total_entries"],
                "critical_operations": critical_count,
                "file_operations": len(file_operations),
                "date_range": summary["date_range"],
                "action_breakdown": summary["action_counts"],
                "compliance_status": "compliant" if summary["total_entries"] > 0 else "no_activity"
            }
            
        except Exception as e:
            print(f"Failed to generate compliance report: {str(e)}")
            return {"error": "Failed to generate compliance report"}
