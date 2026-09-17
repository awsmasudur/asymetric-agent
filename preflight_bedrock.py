"""Read-only Bedrock preflight for us-east-1.

Verifies credentials and lists available inference profiles so you can choose
the right --observer/--decider/--auditor selectors for run.py. Makes NO model
invocations and incurs NO token cost -- it only calls control-plane list APIs.

Usage:
    python preflight_bedrock.py --region us-east-1
    python preflight_bedrock.py --region us-east-1 --match claude
"""

from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser(description="Read-only Bedrock preflight")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--match", default="", help="case-insensitive substring filter")
    args = ap.parse_args()

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        print("boto3 is not installed. Install with: pip install boto3")
        return

    try:
        sts = boto3.client("sts", region_name=args.region)
        ident = sts.get_caller_identity()
        print(f"AWS identity OK: account={ident['Account']} arn={ident['Arn']}")
    except (BotoCoreError, ClientError) as e:
        print(f"Could not verify AWS credentials: {e}")
        print("Configure creds (aws configure / env vars / SSO) and retry.")
        return

    try:
        bedrock = boto3.client("bedrock", region_name=args.region)
        paginator = bedrock.get_paginator("list_inference_profiles")
        found = []
        for page in paginator.paginate():
            for prof in page.get("inferenceProfileSummaries", []):
                name = prof.get("inferenceProfileName", "")
                pid = prof.get("inferenceProfileId", "")
                if args.match and args.match.lower() not in (name + pid).lower():
                    continue
                found.append((name, pid))
    except (BotoCoreError, ClientError) as e:
        print(f"Could not list inference profiles in {args.region}: {e}")
        print("Check that Bedrock is enabled and your role has "
              "bedrock:ListInferenceProfiles.")
        return

    if not found:
        print(f"No inference profiles found in {args.region}"
              + (f" matching '{args.match}'." if args.match else "."))
        return

    print(f"\nInference profiles in {args.region}"
          + (f" matching '{args.match}'" if args.match else "") + ":")
    for name, pid in sorted(found):
        print(f"  - {pid}   ({name})")
    print("\nUse a distinctive substring of the profile id/name as the "
          "--observer/--decider/--auditor selector for run.py.")


if __name__ == "__main__":
    main()
