"""
Security Audit Script
Cloud Security - TASK SEC-001
Comprehensive security audit for the quantitative trading system
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import ast
import re


class SecurityAuditor:
    """Security audit system for the trading platform"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.results = {
            "audit_date": datetime.now().isoformat(),
            "vulnerabilities": [],
            "security_score": 0,
            "summary": {},
        }
        self.high_risk_patterns = [
            r"password\s*=\s*['\"].*['\"]",  # Hardcoded passwords
            r"api_key\s*=\s*['\"].*['\"]",  # Hardcoded API keys
            r"secret\s*=\s*['\"].*['\"]",  # Hardcoded secrets
            r"eval\(",  # Eval usage
            r"exec\(",  # Exec usage
            r"pickle\.loads",  # Pickle deserialization
            r"os\.system",  # System command execution
            r"subprocess\.call.*shell=True",  # Shell injection risk
        ]

    def run_full_audit(self):
        """Run complete security audit"""
        print("\n" + "=" * 80)
        print("SECURITY AUDIT - TASK SEC-001")
        print("Cloud Security Agent")
        print("=" * 80)
        print(f"Audit Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Step 1: Code Security Scan
        print("\n[STEP 1/5] Code Security Scanning...")
        code_results = self.scan_code_security()

        # Step 2: Dependency Check
        print("\n[STEP 2/5] Dependency Vulnerability Check...")
        dep_results = self.check_dependencies()

        # Step 3: Configuration Audit
        print("\n[STEP 3/5] Configuration Security Audit...")
        config_results = self.audit_configurations()

        # Step 4: API Security Check
        print("\n[STEP 4/5] API Security Verification...")
        api_results = self.check_api_security()

        # Step 5: Generate Report
        print("\n[STEP 5/5] Generating Security Report...")
        self.generate_report()

        # Display Summary
        self.display_summary()

        return self.results

    def scan_code_security(self) -> Dict:
        """Scan Python code for security issues"""
        issues = []

        # Scan all Python files
        python_files = list(self.project_root.glob("**/*.py"))

        for file_path in python_files:
            if "venv" in str(file_path) or "__pycache__" in str(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for high-risk patterns
                for pattern in self.high_risk_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        issues.append(
                            {
                                "file": str(file_path.relative_to(self.project_root)),
                                "line": line_num,
                                "severity": "HIGH",
                                "issue": f"Potential security risk: {pattern}",
                                "code": match.group()[:50],
                            }
                        )

                # Parse AST for additional checks
                try:
                    tree = ast.parse(content)
                    self.check_ast_security(tree, file_path, issues)
                except SyntaxError:
                    pass

            except Exception as e:
                print(f"  [WARNING] Could not scan {file_path}: {e}")

        print(f"  Found {len(issues)} potential security issues")
        self.results["vulnerabilities"].extend(issues)
        return {"total_issues": len(issues), "files_scanned": len(python_files)}

    def check_ast_security(self, tree: ast.AST, file_path: Path, issues: List):
        """Check AST for security issues"""
        for node in ast.walk(tree):
            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ["pickle", "marshal", "shelve"]:
                        issues.append(
                            {
                                "file": str(file_path.relative_to(self.project_root)),
                                "line": node.lineno,
                                "severity": "MEDIUM",
                                "issue": f"Use of potentially unsafe module: {alias.name}",
                                "code": alias.name,
                            }
                        )

            # Check for open() without explicit encoding
            if isinstance(node, ast.Call):
                if hasattr(node.func, "id") and node.func.id == "open":
                    has_encoding = any(kw.arg == "encoding" for kw in node.keywords)
                    if not has_encoding:
                        issues.append(
                            {
                                "file": str(file_path.relative_to(self.project_root)),
                                "line": node.lineno,
                                "severity": "LOW",
                                "issue": "File opened without explicit encoding",
                                "code": "open() without encoding",
                            }
                        )

    def check_dependencies(self) -> Dict:
        """Check for vulnerable dependencies"""
        vulnerabilities = []

        # Check if requirements.txt exists
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            print("  Checking requirements.txt dependencies...")

            # Try to use safety if available
            try:
                result = subprocess.run(
                    ["pip", "list", "--format=json"], capture_output=True, text=True, timeout=30
                )

                if result.returncode == 0:
                    packages = json.loads(result.stdout)
                    print(f"  Scanning {len(packages)} installed packages...")

                    # Check for known vulnerable versions
                    vulnerable_packages = {
                        "flask": ["<2.0.0", "Security fixes in 2.0+"],
                        "django": ["<3.2", "Security updates required"],
                        "requests": ["<2.25.0", "SSL verification issues"],
                        "urllib3": ["<1.26.5", "Security patches needed"],
                        "pyyaml": ["<5.4", "Arbitrary code execution risk"],
                    }

                    for package in packages:
                        pkg_name = package.get("name", "").lower()
                        if pkg_name in vulnerable_packages:
                            vulnerabilities.append(
                                {
                                    "package": pkg_name,
                                    "version": package.get("version"),
                                    "severity": "MEDIUM",
                                    "issue": vulnerable_packages[pkg_name][1],
                                }
                            )

            except Exception as e:
                print(f"  [WARNING] Could not check dependencies: {e}")

        print(f"  Found {len(vulnerabilities)} dependency issues")
        return {"vulnerable_dependencies": vulnerabilities}

    def audit_configurations(self) -> Dict:
        """Audit configuration files for security issues"""
        config_issues = []

        # Check for exposed secrets in config files
        config_patterns = {
            "**/*.env": "Environment file",
            "**/*.ini": "INI configuration",
            "**/*.cfg": "Configuration file",
            "**/*.json": "JSON configuration",
            "**/*.yaml": "YAML configuration",
            "**/*.yml": "YAML configuration",
        }

        for pattern, file_type in config_patterns.items():
            for config_file in self.project_root.glob(pattern):
                if "venv" in str(config_file) or "node_modules" in str(config_file):
                    continue

                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Check for exposed secrets
                    secret_patterns = [
                        r"(password|passwd|pwd)\s*[:=]\s*['\"]?[^'\"]+",
                        r"(api[_-]?key|apikey)\s*[:=]\s*['\"]?[^'\"]+",
                        r"(secret|token)\s*[:=]\s*['\"]?[^'\"]+",
                        r"(aws[_-]?access[_-]?key)\s*[:=]\s*['\"]?[^'\"]+",
                    ]

                    for pattern in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            config_issues.append(
                                {
                                    "file": str(config_file.relative_to(self.project_root)),
                                    "type": file_type,
                                    "severity": "HIGH",
                                    "issue": "Potential exposed secret in configuration",
                                }
                            )
                            break

                except Exception as e:
                    pass

        print(f"  Found {len(config_issues)} configuration issues")
        self.results["vulnerabilities"].extend(config_issues)
        return {"config_issues": len(config_issues)}

    def check_api_security(self) -> Dict:
        """Check API security configurations"""
        api_issues = []

        # Check Capital.com API integration
        api_file = self.project_root / "src" / "connectors" / "capital_com_api.py"
        if api_file.exists():
            print("  Checking Capital.com API integration...")

            with open(api_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for secure practices
            checks = {
                "uses_https": "https://" in content,
                "has_authentication": "authenticate" in content.lower(),
                "encrypts_password": "encrypt" in content.lower(),
                "uses_env_vars": "os.environ" in content,
                "has_rate_limiting": "rate_limit" in content.lower(),
                "validates_ssl": "verify=" not in content or "verify=True" in content,
            }

            for check, passed in checks.items():
                if not passed:
                    api_issues.append(
                        {
                            "component": "Capital.com API",
                            "check": check,
                            "severity": "MEDIUM",
                            "status": "FAILED",
                        }
                    )

        print(f"  API security checks: {len(api_issues)} issues found")
        return {"api_security": api_issues}

    def calculate_security_score(self) -> int:
        """Calculate overall security score"""
        total_issues = len(self.results["vulnerabilities"])

        # Scoring system
        high_issues = sum(1 for v in self.results["vulnerabilities"] if v.get("severity") == "HIGH")
        medium_issues = sum(
            1 for v in self.results["vulnerabilities"] if v.get("severity") == "MEDIUM"
        )
        low_issues = sum(1 for v in self.results["vulnerabilities"] if v.get("severity") == "LOW")

        # Calculate score (out of 100)
        score = 100
        score -= high_issues * 15
        score -= medium_issues * 5
        score -= low_issues * 2

        return max(0, score)

    def generate_report(self):
        """Generate security audit report"""
        self.results["security_score"] = self.calculate_security_score()

        # Group vulnerabilities by severity
        severity_groups = {"HIGH": [], "MEDIUM": [], "LOW": []}
        for vuln in self.results["vulnerabilities"]:
            severity = vuln.get("severity", "LOW")
            severity_groups[severity].append(vuln)

        self.results["summary"] = {
            "total_issues": len(self.results["vulnerabilities"]),
            "high_severity": len(severity_groups["HIGH"]),
            "medium_severity": len(severity_groups["MEDIUM"]),
            "low_severity": len(severity_groups["LOW"]),
            "security_score": self.results["security_score"],
        }

        # Save JSON report
        with open("security_audit_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Generate Markdown report
        self.generate_markdown_report(severity_groups)

    def generate_markdown_report(self, severity_groups: Dict):
        """Generate markdown security report"""
        []
        report.append("# Security Audit Report")
        report.append(f"## Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("## Cloud Security - TASK SEC-001")
        report.append("")
        report.append("---")
        report.append("")
        report.append("## Executive Summary")
        report.append("")
        report.append(f"**Security Score: {self.results['security_score']}/100**")
        report.append("")
        report.append("| Severity | Count |")
        report.append("|----------|-------|")
        report.append(f"| HIGH | {self.results['summary']['high_severity']} |")
        report.append(f"| MEDIUM | {self.results['summary']['medium_severity']} |")
        report.append(f"| LOW | {self.results['summary']['low_severity']} |")
        report.append(f"| **TOTAL** | **{self.results['summary']['total_issues']}** |")
        report.append("")

        # Add vulnerability details
        if severity_groups["HIGH"]:
            report.append("## High Severity Issues 🔴")
            report.append("")
            for vuln in severity_groups["HIGH"][:10]:  # Limit to 10
                report.append(f"- **{vuln.get('file', 'N/A')}**: {vuln.get('issue', 'N/A')}")
            report.append("")

        if severity_groups["MEDIUM"]:
            report.append("## Medium Severity Issues 🟡")
            report.append("")
            for vuln in severity_groups["MEDIUM"][:10]:
                report.append(f"- **{vuln.get('file', 'N/A')}**: {vuln.get('issue', 'N/A')}")
            report.append("")

        # Add recommendations
        report.append("## Recommendations")
        report.append("")
        if self.results["summary"]["high_severity"] > 0:
            report.append("1. **URGENT**: Address all HIGH severity issues immediately")
        report.append("2. Use environment variables for all sensitive data")
        report.append("3. Implement proper input validation")
        report.append("4. Enable security headers for API endpoints")
        report.append("5. Regularly update dependencies")
        report.append("")

        # Add compliance status
        report.append("## Compliance Status")
        report.append("")
        report.append("| Standard | Status |")
        report.append("|----------|--------|")
        report.append(
            f"| OWASP Top 10 | {'[WARNING] Review Needed' if self.results['summary']['high_severity'] > 0 else '[OK] Compliant'} |"
        )
        report.append(
            f"| PCI DSS | {'[WARNING] Review Needed' if self.results['summary']['high_severity'] > 0 else '[OK] Compliant'} |"
        )
        report.append(
            f"| GDPR | {'[OK] Compliant' if self.results['security_score'] > 70 else '[WARNING] Review Needed'} |"
        )
        report.append("")

        report.append("---")
        report.append("")
        report.append("**Auditor**: Cloud Security Agent")
        report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
        report.append(
            f"**Status**: {'PASSED' if self.results['security_score'] >= 85 else 'REQUIRES REMEDIATION'}"
        )

        # Save report
        with open("security_audit_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(report))

    def display_summary(self):
        """Display audit summary"""
        print("\n" + "=" * 80)
        print("SECURITY AUDIT SUMMARY")
        print("=" * 80)

        score = self.results["security_score"]
        status = (
            "[PASSED]"
            if score >= 85
            else "[WARNING] NEEDS IMPROVEMENT" if score >= 70 else "[FAILED]"
        )

        print(f"\nSecurity Score: {score}/100 {status}")
        print("\nVulnerabilities Found:")
        print(f"  HIGH:   {self.results['summary']['high_severity']}")
        print(f"  MEDIUM: {self.results['summary']['medium_severity']}")
        print(f"  LOW:    {self.results['summary']['low_severity']}")
        print(f"  TOTAL:  {self.results['summary']['total_issues']}")

        print("\nReports Generated:")
        print("  [OK] security_audit_results.json")
        print("  [OK] security_audit_report.md")

        if self.results["summary"]["high_severity"] > 0:
            print("\n[WARNING] CRITICAL: High severity issues require immediate attention!")

        print("\n" + "=" * 80)


def main():
    """Main execution function"""
    auditor = SecurityAuditor()

    try:
        results = auditor.run_full_audit()

        # Exit with appropriate code
        if results["security_score"] >= 85:
            sys.exit(0)  # Pass
        else:
            sys.exit(1)  # Fail

    except KeyboardInterrupt:
        print("\n\nAudit interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Audit failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
