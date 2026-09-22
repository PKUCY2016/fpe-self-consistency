import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Periodic FPE probability flows; empirical evidence only")
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("doctor")
    for command in ("train","evolve"):
        p = sub.add_parser(command)
        p.add_argument("--problem",default="heat1d")
        p.add_argument("--problem-json",type=Path)
        p.add_argument("--config",type=Path)
        p.add_argument("--run",type=Path,required=True)
        p.add_argument("--seed",type=int,default=0)
        for name,typ in (("updates",int),("steps",int),("batch",int),("max-seconds",float),
                         ("max-rounds",int),("validation-samples",int),("K",int),("p",int)):
            p.add_argument("--"+name,type=typ)
    p = sub.add_parser("resume")
    p.add_argument("--run",type=Path,required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--grid-size",type=int)
    p.add_argument("--output",type=Path)
    args = parser.parse_args(argv)
    from .runner import create_run, environment, execute, get_model
    from .storage import read_json, write_json
    if args.command == "doctor":
        print(json.dumps(environment(),indent=2))
        return 0
    if args.command in ("train","evolve"):
        from .config import SolverConfig
        from .problems import Problem, benchmark
        problem = Problem(**read_json(args.problem_json)) if args.problem_json else benchmark(args.problem)
        config = SolverConfig(**read_json(args.config)) if args.config else SolverConfig()
        override = {k:getattr(args,k) for k in ("updates","steps","batch","max_seconds","max_rounds",
                    "validation_samples","K","p") if getattr(args,k) is not None}
        config = replace(config,**override)
        create_run(args.run,problem,config,args.seed,args.command)
        state = execute(args.run)
    elif args.command == "resume":
        state = execute(args.run,resume=True)
    else:
        from .evaluation import evaluate_model
        basis,coeff,problem = get_model(args.run)
        report = evaluate_model(basis,coeff,problem,grid_size=args.grid_size)
        report["checkpoint"] = read_json(args.run/"state.json")["champion"]["checkpoint"]
        output = args.output or args.run/"evaluation.json"
        write_json(output,report)
        print(json.dumps({"report":str(output),"result":report},indent=2))
        return 0
    print(json.dumps({"run":str(args.run),"status":state["status"],"rounds":len(state["events"]),
                      "elapsed_seconds":state["elapsed_seconds"],"evidence_type":"empirical"},indent=2))
    return 2 if state["status"] in ("NUMERICAL_FAILURE","INTERRUPTED") else 0


if __name__ == "__main__":
    sys.exit(main())
