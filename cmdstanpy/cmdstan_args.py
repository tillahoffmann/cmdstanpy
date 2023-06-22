"""
CmdStan arguments
"""
import os
from enum import Enum, auto
from time import time
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
from numpy.random import RandomState

from cmdstanpy import _TMPDIR
from cmdstanpy.utils import (
    cmdstan_path,
    cmdstan_version_before,
    create_named_text_file,
    get_logger,
    read_metric,
    write_stan_json,
)

OptionalPath = Union[str, os.PathLike, None]


class Method(Enum):
    """Supported CmdStan method names."""

    SAMPLE = auto()
    OPTIMIZE = auto()
    GENERATE_QUANTITIES = auto()
    VARIATIONAL = auto()

    def __repr__(self) -> str:
        return '<{self.__class__.__name__}.{self.name}}>'


class SamplerArgs:
    """Arguments for the NUTS adaptive sampler."""

    def __init__(
        self,
        iter_warmup: Optional[int] = None,
        iter_sampling: Optional[int] = None,
        save_warmup: bool = False,
        thin: Optional[int] = None,
        max_treedepth: Optional[int] = None,
        metric: Union[
            str, Dict[str, Any], List[str], List[Dict[str, Any]], None
        ] = None,
        step_size: Union[float, List[float], None] = None,
        adapt_engaged: bool = True,
        adapt_delta: Optional[float] = None,
        adapt_init_phase: Optional[int] = None,
        adapt_metric_window: Optional[int] = None,
        adapt_step_size: Optional[int] = None,
        fixed_param: bool = False,
    ) -> None:
        """Initialize object."""
        self.iter_warmup = iter_warmup
        self.iter_sampling = iter_sampling
        self.save_warmup = save_warmup
        self.thin = thin
        self.max_treedepth = max_treedepth
        self.metric = metric
        self.metric_type: Optional[str] = None
        self.metric_file: Union[str, List[str], None] = None
        self.step_size = step_size
        self.adapt_engaged = adapt_engaged
        self.adapt_delta = adapt_delta
        self.adapt_init_phase = adapt_init_phase
        self.adapt_metric_window = adapt_metric_window
        self.adapt_step_size = adapt_step_size
        self.fixed_param = fixed_param
        self.diagnostic_file = None

    def validate(self, chains: Optional[int]) -> None:
        """
        Check arguments correctness and consistency.

        * adaptation and warmup args are consistent
        * if file(s) for metric are supplied, check contents.
        * length of per-chain lists equals specified # of chains
        """
        if not isinstance(chains, (int, np.integer)) or chains < 1:
            raise ValueError(
                'Sampler expects number of chains to be greater than 0.'
            )
        if not (
            self.adapt_delta is None
            and self.adapt_init_phase is None
            and self.adapt_metric_window is None
            and self.adapt_step_size is None
        ):
            if self.adapt_engaged is False:
                parts = ['Conflicting arguments: adapt_engaged: False']
                keys = ['adapt_delta', 'adapt_init_phase',
                        'adapt_metric_window', 'adapt_step_size']
                for key in keys:
                    parts.append(f'{key}: {getattr(self, key)}')
                raise ValueError(', '.join(parts))

        if self.iter_warmup is not None:
            if self.iter_warmup < 0 or not isinstance(
                self.iter_warmup, (int, np.integer)
            ):
                raise ValueError(
                    'Value for iter_warmup must be a non-negative integer,'
                    f' found {self.iter_warmup}.'
                )
            if self.iter_warmup > 0 and not self.adapt_engaged:
                raise ValueError(
                    'Argument "adapt_engaged" is False, '
                    'cannot specify warmup iterations.'
                )
        if self.iter_sampling is not None:
            if self.iter_sampling < 0 or not isinstance(
                self.iter_sampling, (int, np.integer)
            ):
                raise ValueError(
                    'Argument "iter_sampling" must be a non-negative integer,'
                    f' found {self.iter_sampling}.'
                )
        if self.thin is not None:
            if self.thin < 1 or not isinstance(self.thin, (int, np.integer)):
                raise ValueError(
                    'Argument "thin" must be a positive integer,'
                    f'found {self.thin}.'
                )
        if self.max_treedepth is not None:
            if self.max_treedepth < 1 or not isinstance(
                self.max_treedepth, (int, np.integer)
            ):
                raise ValueError(
                    'Argument "max_treedepth" must be a positive integer,'
                    f' found {self.max_treedepth}.'
                )
        if self.step_size is not None:
            if isinstance(
                self.step_size, (float, int, np.integer, np.floating)
            ):
                if self.step_size <= 0:
                    raise ValueError(
                        'Argument "step_size" must be > 0, '
                        f'found {self.step_size}.'
                    )
            else:
                if len(self.step_size) != chains:
                    raise ValueError(
                        f'Expecting {chains} per-chain step_size '
                        f'specifications, found {len(self.step_size)}.'
                    )
                for i, step_size in enumerate(self.step_size):
                    if step_size < 0:
                        raise ValueError(
                            'Argument "step_size" must be > 0, '
                            f'chain {i + 1}, found {step_size}.'
                        )
        if self.metric is not None:
            if isinstance(self.metric, str):
                if self.metric in ['diag', 'diag_e']:
                    self.metric_type = 'diag_e'
                elif self.metric in ['dense', 'dense_e']:
                    self.metric_type = 'dense_e'
                elif self.metric in ['unit', 'unit_e']:
                    self.metric_type = 'unit_e'
                else:
                    if not os.path.exists(self.metric):
                        raise ValueError(f'no such file {self.metric}')
                    dims = read_metric(self.metric)
                    if len(dims) == 1:
                        self.metric_type = 'diag_e'
                    else:
                        self.metric_type = 'dense_e'
                    self.metric_file = self.metric
            elif isinstance(self.metric, dict):
                if 'inv_metric' not in self.metric:
                    raise ValueError(
                        'Entry "inv_metric" not found in metric dict.'
                    )
                dims = list(np.asarray(self.metric['inv_metric']).shape)
                if len(dims) == 1:
                    self.metric_type = 'diag_e'
                else:
                    self.metric_type = 'dense_e'
                dict_file = create_named_text_file(
                    dir=_TMPDIR, prefix="metric", suffix=".json"
                )
                write_stan_json(dict_file, self.metric)
                self.metric_file = dict_file
            elif isinstance(self.metric, (list, tuple)):
                if len(self.metric) != chains:
                    raise ValueError(
                        'Number of metric files must match number of chains, '
                        f'found {len(self.metric)} metric files for {chains} '
                        'chains.'
                    )
                if all(isinstance(elem, dict) for elem in self.metric):
                    metric_files: List[str] = []
                    for i, metric in enumerate(self.metric):
                        metric_dict: Dict[str, Any] = metric  # type: ignore
                        if 'inv_metric' not in metric_dict:
                            raise ValueError(
                                'Entry "inv_metric" not found in metric dict '
                                f'for chain {i + 1}.'
                            )
                        if i == 0:
                            dims = list(
                                np.asarray(metric_dict['inv_metric']).shape
                            )
                        else:
                            dims2 = list(
                                np.asarray(metric_dict['inv_metric']).shape
                            )
                            if dims != dims2:
                                raise ValueError(
                                    'Found inconsistent "inv_metric" entry '
                                    f'for chain {i + 1}: entry has dims '
                                    f'{dims}, expected {dims2}.'
                                )
                        dict_file = create_named_text_file(
                            dir=_TMPDIR, prefix="metric", suffix=".json"
                        )
                        write_stan_json(dict_file, metric_dict)
                        metric_files.append(dict_file)
                    if len(dims) == 1:
                        self.metric_type = 'diag_e'
                    else:
                        self.metric_type = 'dense_e'
                    self.metric_file = metric_files
                elif all(isinstance(elem, str) for elem in self.metric):
                    metric_files = []
                    for i, metric in enumerate(self.metric):
                        assert isinstance(metric, str)  # typecheck
                        if not os.path.exists(metric):
                            raise ValueError('no such file {metric}')
                        if i == 0:
                            dims = read_metric(metric)
                        else:
                            dims2 = read_metric(metric)
                            if len(dims) != len(dims2):
                                raise ValueError(
                                    f'Metrics files {self.metric[0]}, '
                                    f'{metric}, inconsistent metrics'
                                )
                            if dims != dims2:
                                raise ValueError(
                                    f'Metrics files {self.metric[0]}, '
                                    f'{metric}, inconsistent metrics'
                                )
                        metric_files.append(metric)
                    if len(dims) == 1:
                        self.metric_type = 'diag_e'
                    else:
                        self.metric_type = 'dense_e'
                    self.metric_file = metric_files
                else:
                    raise ValueError(
                        'Argument "metric" must be a list of pathnames or '
                        f'Python dicts, found list of {type(self.metric[0])}.'
                    )
            else:
                raise ValueError(
                    'Invalid metric specified, not a recognized metric type, '
                    'must be either a metric type name, a filepath, dict, '
                    'or list of per-chain filepaths or dicts.  Found '
                    f'an object of type {type(self.metric)}.'
                )

        if self.adapt_delta is not None:
            if not 0 < self.adapt_delta < 1:
                raise ValueError(
                    'Argument "adapt_delta" must be between 0 and 1, '
                    f'found {self.adapt_delta}'
                )
        if self.adapt_init_phase is not None:
            if self.adapt_init_phase < 0 or not isinstance(
                self.adapt_init_phase, (int, np.integer)
            ):
                raise ValueError(
                    'Argument "adapt_init_phase" must be a non-negative '
                    f'integer, found {self.adapt_init_phase}'
                )
        if self.adapt_metric_window is not None:
            if self.adapt_metric_window < 0 or not isinstance(
                self.adapt_metric_window, (int, np.integer)
            ):
                raise ValueError(
                    'Argument "adapt_metric_window" must be a non-negative '
                    f'integer, found {self.adapt_metric_window}'
                )
        if self.adapt_step_size is not None:
            if self.adapt_step_size < 0 or not isinstance(
                self.adapt_step_size, (int, np.integer)
            ):
                raise ValueError(
                    'Argument "adapt_step_size" must be a non-negative '
                    f'integer, found {self.adapt_step_size}'
                )

        if self.fixed_param and (
            self.max_treedepth is not None
            or self.metric is not None
            or self.step_size is not None
            or not (
                self.adapt_delta is None
                and self.adapt_init_phase is None
                and self.adapt_metric_window is None
                and self.adapt_step_size is None
            )
        ):
            raise ValueError(
                'When fixed_param=True, cannot specify adaptation parameters.'
            )

    def compose(self, idx: int, cmd: List[str]) -> List[str]:
        """
        Compose CmdStan command for method-specific non-default arguments.
        """
        cmd.append('method=sample')
        if self.iter_sampling is not None:
            cmd.append(f'num_samples={self.iter_sampling}')
        if self.iter_warmup is not None:
            cmd.append(f'num_warmup={self.iter_warmup}')
        if self.save_warmup:
            cmd.append('save_warmup=1')
        if self.thin is not None:
            cmd.append(f'thin={self.thin}')
        if self.fixed_param:
            cmd.append('algorithm=fixed_param')
            return cmd
        else:
            cmd.append('algorithm=hmc')
        if self.max_treedepth is not None:
            cmd.append('engine=nuts')
            cmd.append(f'max_depth={self.max_treedepth}')
        if self.step_size is not None:
            if not isinstance(self.step_size, list):
                cmd.append(f'stepsize={self.step_size}')
            else:
                cmd.append(f'stepsize={self.step_size[idx]}')
        if self.metric is not None:
            cmd.append(f'metric={self.metric_type}')
        if self.metric_file is not None:
            if not isinstance(self.metric_file, list):
                cmd.append(f'metric_file={self.metric_file}')
            else:
                cmd.append(f'metric_file={self.metric_file[idx]}')
        cmd.append('adapt')
        if self.adapt_engaged:
            cmd.append('engaged=1')
        else:
            cmd.append('engaged=0')
        if self.adapt_delta is not None:
            cmd.append(f'delta={self.adapt_delta}')
        if self.adapt_init_phase is not None:
            cmd.append(f'init_buffer={self.adapt_init_phase}')
        if self.adapt_metric_window is not None:
            cmd.append(f'window={self.adapt_metric_window}')
        if self.adapt_step_size is not None:
            cmd.append(f'term_buffer={self.adapt_step_size}')

        return cmd


class OptimizeArgs:
    """Container for arguments for the optimizer."""

    OPTIMIZE_ALGOS = {'BFGS', 'bfgs', 'LBFGS', 'lbfgs', 'Newton', 'newton'}

    def __init__(
        self,
        algorithm: Optional[str] = None,
        init_alpha: Optional[float] = None,
        iter: Optional[int] = None,
        save_iterations: bool = False,
        tol_obj: Optional[float] = None,
        tol_rel_obj: Optional[float] = None,
        tol_grad: Optional[float] = None,
        tol_rel_grad: Optional[float] = None,
        tol_param: Optional[float] = None,
        history_size: Optional[int] = None,
    ) -> None:

        self.algorithm = algorithm or ""
        self.init_alpha = init_alpha
        self.iter = iter
        self.save_iterations = save_iterations
        self.tol_obj = tol_obj
        self.tol_rel_obj = tol_rel_obj
        self.tol_grad = tol_grad
        self.tol_rel_grad = tol_rel_grad
        self.tol_param = tol_param
        self.history_size = history_size
        self.thin = None

    def validate(
        self, chains: Optional[int] = None  # pylint: disable=unused-argument
    ) -> None:
        """
        Check arguments correctness and consistency.
        """
        if self.algorithm and self.algorithm not in self.OPTIMIZE_ALGOS:
            raise ValueError(
                'Please specify optimizer algorithms as one of '
                f'[{", ".join(self.OPTIMIZE_ALGOS)}]'
            )

        if self.init_alpha is not None:
            if self.algorithm.lower() not in {'lbfgs', 'bfgs'}:
                raise ValueError(
                    'init_alpha requires that algorithm be set to bfgs or lbfgs'
                )
            if isinstance(self.init_alpha, (float, np.floating)):
                if self.init_alpha <= 0:
                    raise ValueError('init_alpha must be greater than 0')
            else:
                raise ValueError('init_alpha must be type of float')

        if self.iter is not None:
            if isinstance(self.iter, (int, np.integer)):
                if self.iter < 0:
                    raise ValueError('iter must be greater than 0')
            else:
                raise ValueError('iter must be type of int')

        if self.tol_obj is not None:
            if self.algorithm.lower() not in {'lbfgs', 'bfgs'}:
                raise ValueError(
                    'tol_obj requires that algorithm be set to bfgs or lbfgs'
                )
            if isinstance(self.tol_obj, (float, np.floating)):
                if self.tol_obj <= 0:
                    raise ValueError('tol_obj must be greater than 0')
            else:
                raise ValueError('tol_obj must be type of float')

        if self.tol_rel_obj is not None:
            if self.algorithm.lower() not in {'lbfgs', 'bfgs'}:
                raise ValueError(
                    'tol_rel_obj requires that algorithm be set to bfgs'
                    ' or lbfgs'
                )
            if isinstance(self.tol_rel_obj, (float, np.floating)):
                if self.tol_rel_obj <= 0:
                    raise ValueError('tol_rel_obj must be greater than 0')
            else:
                raise ValueError('tol_rel_obj must be type of float')

        if self.tol_grad is not None:
            if self.algorithm.lower() not in {'lbfgs', 'bfgs'}:
                raise ValueError(
                    'tol_grad requires that algorithm be set to bfgs or lbfgs'
                )
            if isinstance(self.tol_grad, (float, np.floating)):
                if self.tol_grad <= 0:
                    raise ValueError('tol_grad must be greater than 0')
            else:
                raise ValueError('tol_grad must be type of float')

        if self.tol_rel_grad is not None:
            if self.algorithm.lower() not in {'lbfgs', 'bfgs'}:
                raise ValueError(
                    'tol_rel_grad requires that algorithm be set to bfgs'
                    ' or lbfgs'
                )
            if isinstance(self.tol_rel_grad, (float, np.floating)):
                if self.tol_rel_grad <= 0:
                    raise ValueError('tol_rel_grad must be greater than 0')
            else:
                raise ValueError('tol_rel_grad must be type of float')

        if self.tol_param is not None:
            if self.algorithm.lower() not in {'lbfgs', 'bfgs'}:
                raise ValueError(
                    'tol_param requires that algorithm be set to bfgs or lbfgs'
                )
            if isinstance(self.tol_param, (float, np.floating)):
                if self.tol_param <= 0:
                    raise ValueError('tol_param must be greater than 0')
            else:
                raise ValueError('tol_param must be type of float')

        if self.history_size is not None:
            if self.algorithm.lower() != 'lbfgs':
                raise ValueError(
                    'history_size requires that algorithm be set to lbfgs'
                )
            if isinstance(self.history_size, (int, np.integer)):
                if self.history_size < 0:
                    raise ValueError('history_size must be greater than 0')
            else:
                raise ValueError('history_size must be type of int')

    # pylint: disable=unused-argument
    def compose(self, idx: int, cmd: List[str]) -> List[str]:
        """compose command string for CmdStan for non-default arg values."""
        cmd.append('method=optimize')
        if self.algorithm:
            cmd.append(f'algorithm={self.algorithm.lower()}')
        if self.init_alpha is not None:
            cmd.append(f'init_alpha={self.init_alpha}')
        if self.tol_obj is not None:
            cmd.append(f'tol_obj={self.tol_obj}')
        if self.tol_rel_obj is not None:
            cmd.append(f'tol_rel_obj={self.tol_rel_obj}')
        if self.tol_grad is not None:
            cmd.append(f'tol_grad={self.tol_grad}')
        if self.tol_rel_grad is not None:
            cmd.append(f'tol_rel_grad={self.tol_rel_grad}')
        if self.tol_param is not None:
            cmd.append(f'tol_param={self.tol_param}')
        if self.history_size is not None:
            cmd.append(f'history_size={self.history_size}')
        if self.iter is not None:
            cmd.append(f'iter={self.iter}')
        if self.save_iterations:
            cmd.append('save_iterations=1')

        return cmd


class GenerateQuantitiesArgs:
    """Arguments needed for generate_quantities method."""

    def __init__(self, csv_files: List[str]) -> None:
        """Initialize object."""
        self.sample_csv_files = csv_files

    def validate(
        self, chains: Optional[int] = None  # pylint: disable=unused-argument
    ) -> None:
        """
        Check arguments correctness and consistency.

        * check that sample csv files exist
        """
        for csv in self.sample_csv_files:
            if not os.path.exists(csv):
                raise ValueError(
                    f'Invalid path for sample csv file: {csv}'
                )

    def compose(self, idx: int, cmd: List[str]) -> List[str]:
        """
        Compose CmdStan command for method-specific non-default arguments.
        """
        cmd.append('method=generate_quantities')
        cmd.append(f'fitted_params={self.sample_csv_files[idx]}')
        return cmd


class VariationalArgs:
    """Arguments needed for variational method."""

    VARIATIONAL_ALGOS = {'meanfield', 'fullrank'}

    def __init__(
        self,
        algorithm: Optional[str] = None,
        iter: Optional[int] = None,
        grad_samples: Optional[int] = None,
        elbo_samples: Optional[int] = None,
        eta: Optional[float] = None,
        adapt_iter: Optional[int] = None,
        adapt_engaged: bool = True,
        tol_rel_obj: Optional[float] = None,
        eval_elbo: Optional[int] = None,
        output_samples: Optional[int] = None,
    ) -> None:
        self.algorithm = algorithm
        self.iter = iter
        self.grad_samples = grad_samples
        self.elbo_samples = elbo_samples
        self.eta = eta
        self.adapt_iter = adapt_iter
        self.adapt_engaged = adapt_engaged
        self.tol_rel_obj = tol_rel_obj
        self.eval_elbo = eval_elbo
        self.output_samples = output_samples

    def validate(
        self, chains: Optional[int] = None  # pylint: disable=unused-argument
    ) -> None:
        """
        Check arguments correctness and consistency.
        """
        if (
            self.algorithm is not None
            and self.algorithm not in self.VARIATIONAL_ALGOS
        ):
            raise ValueError(
                'Please specify variational algorithms as one of '
                f'[{", ".join(self.VARIATIONAL_ALGOS)}]'
            )
        if self.iter is not None:
            if self.iter < 1 or not isinstance(self.iter, (int, np.integer)):
                raise ValueError(
                    f'iter must be a positive integer, found {self.iter}'
                )
        if self.grad_samples is not None:
            if self.grad_samples < 1 or not isinstance(
                self.grad_samples, (int, np.integer)
            ):
                raise ValueError(
                    'grad_samples must be a positive integer, found '
                    f'{self.grad_samples}'
                )
        if self.elbo_samples is not None:
            if self.elbo_samples < 1 or not isinstance(
                self.elbo_samples, (int, np.integer)
            ):
                raise ValueError(
                    'elbo_samples must be a positive integer, found'
                    f'{self.elbo_samples}'
                )
        if self.eta is not None:
            if self.eta < 0 or not isinstance(
                self.eta, (int, float, np.integer, np.floating)
            ):
                raise ValueError(
                    f'eta must be a non-negative number, found {self.eta}'
                )
        if self.adapt_iter is not None:
            if self.adapt_iter < 1 or not isinstance(
                self.adapt_iter, (int, np.integer)
            ):
                raise ValueError(
                    'adapt_iter must be a positive integer, found '
                    f'{self.adapt_iter}'
                )
        if self.tol_rel_obj is not None:
            if self.tol_rel_obj <= 0 or not isinstance(
                self.tol_rel_obj, (int, float, np.integer, np.floating)
            ):
                raise ValueError(
                    'tol_rel_obj must be a positive number, found '
                    f'{self.tol_rel_obj}'
                )
        if self.eval_elbo is not None:
            if self.eval_elbo < 1 or not isinstance(
                self.eval_elbo, (int, np.integer)
            ):
                raise ValueError(
                    'eval_elbo must be a positive integer, found '
                    f'{self.eval_elbo}'
                )
        if self.output_samples is not None:
            if self.output_samples < 1 or not isinstance(
                self.output_samples, (int, np.integer)
            ):
                raise ValueError(
                    'output_samples must be a positive integer, found'
                    f'{self.output_samples}'
                )

    # pylint: disable=unused-argument
    def compose(self, idx: int, cmd: List[str]) -> List[str]:
        """
        Compose CmdStan command for method-specific non-default arguments.
        """
        cmd.append('method=variational')
        if self.algorithm is not None:
            cmd.append(f'algorithm={self.algorithm}')
        if self.iter is not None:
            cmd.append(f'iter={self.iter}')
        if self.grad_samples is not None:
            cmd.append(f'grad_samples={self.grad_samples}')
        if self.elbo_samples is not None:
            cmd.append(f'elbo_samples={self.elbo_samples}')
        if self.eta is not None:
            cmd.append(f'eta={self.eta}')
        cmd.append('adapt')
        if self.adapt_engaged:
            cmd.append('engaged=1')
            if self.adapt_iter is not None:
                cmd.append(f'iter={self.adapt_iter}')
        else:
            cmd.append('engaged=0')
        if self.tol_rel_obj is not None:
            cmd.append(f'tol_rel_obj={self.tol_rel_obj}')
        if self.eval_elbo is not None:
            cmd.append(f'eval_elbo={self.eval_elbo}')
        if self.output_samples is not None:
            cmd.append(f'output_samples={self.output_samples}')
        return cmd


class CmdStanArgs:
    """
    Container for CmdStan command line arguments.
    Consists of arguments common to all methods and
    and an object which contains the method-specific arguments.
    """

    def __init__(
        self,
        model_name: str,
        model_exe: OptionalPath,
        chain_ids: Optional[List[int]],
        method_args: Union[
            SamplerArgs, OptimizeArgs, GenerateQuantitiesArgs, VariationalArgs
        ],
        data: Union[Mapping[str, Any], str, None] = None,
        seed: Union[int, List[int], None] = None,
        inits: Union[int, float, str, List[str], None] = None,
        output_dir: OptionalPath = None,
        sig_figs: Optional[int] = None,
        save_latent_dynamics: bool = False,
        save_profile: bool = False,
        refresh: Optional[int] = None,
    ) -> None:
        """Initialize object."""
        self.model_name = model_name
        self.model_exe = model_exe
        self.chain_ids = chain_ids
        self.data = data
        self.seed = seed
        self.inits = inits
        self.output_dir = output_dir
        self.sig_figs = sig_figs
        self.save_latent_dynamics = save_latent_dynamics
        self.save_profile = save_profile
        self.refresh = refresh
        self.method_args = method_args
        if isinstance(method_args, SamplerArgs):
            self.method = Method.SAMPLE
        elif isinstance(method_args, OptimizeArgs):
            self.method = Method.OPTIMIZE
        elif isinstance(method_args, GenerateQuantitiesArgs):
            self.method = Method.GENERATE_QUANTITIES
        elif isinstance(method_args, VariationalArgs):
            self.method = Method.VARIATIONAL
        self.method_args.validate(len(chain_ids) if chain_ids else None)
        self.validate()

    def validate(self) -> None:
        """
        Check arguments correctness and consistency.

        * input files must exist
        * output files must be in a writeable directory
        * if no seed specified, set random seed.
        * length of per-chain lists equals specified # of chains
        """
        if self.model_name is None:
            raise ValueError('no stan model specified')
        if self.model_exe is None:
            raise ValueError('model not compiled')

        if self.chain_ids is not None:
            for chain_id in self.chain_ids:
                if chain_id < 1:
                    raise ValueError(f'invalid chain_id {chain_id}')
        if self.output_dir is not None:
            self.output_dir = os.path.realpath(
                os.path.expanduser(self.output_dir)
            )
            if not os.path.exists(self.output_dir):
                try:
                    os.makedirs(self.output_dir)
                    get_logger().info(
                        'created output directory: %s', self.output_dir
                    )
                except (RuntimeError, PermissionError) as exc:
                    raise ValueError(
                        'Invalid path for output files, no such dir: '
                        f'{self.output_dir}.'
                    ) from exc
            if not os.path.isdir(self.output_dir):
                raise ValueError(
                    'Specified output_dir is not a directory: '
                    f'{self.output_dir}.'
                )
            try:
                testpath = os.path.join(self.output_dir, str(time()))
                with open(testpath, 'w+'):
                    pass
                os.remove(testpath)  # cleanup
            except Exception as exc:
                raise ValueError(
                    'Invalid path for output files, cannot write to directory: '
                    f'{self.output_dir}'
                ) from exc
        if self.refresh is not None:
            if (
                not isinstance(self.refresh, (int, np.integer))
                or self.refresh < 1
            ):
                raise ValueError(
                    'Argument "refresh" must be a positive integer value, '
                    f'found {self.refresh}.'
                )

        if self.sig_figs is not None:
            if (
                not isinstance(self.sig_figs, (int, np.integer))
                or self.sig_figs < 1
                or self.sig_figs > 18
            ):
                raise ValueError(
                    'Argument "sig_figs" must be an integer between 1 and 18, '
                    f'found {self.sig_figs}'
                )
            # TODO: remove at some future release
            if cmdstan_version_before(2, 25):
                self.sig_figs = None
                get_logger().warning(
                    'Argument "sig_figs" invalid for CmdStan versions < 2.25, '
                    'using version %s in directory %s',
                    os.path.basename(cmdstan_path()),
                    os.path.dirname(cmdstan_path()),
                )

        if self.seed is None:
            rng = RandomState()
            self.seed = rng.randint(1, 99999 + 1)
        else:
            if not isinstance(self.seed, (int, list, np.integer)):
                raise ValueError(
                    'Argument "seed" must be an integer between 0 and 2**32-1, '
                    f'found {self.seed}.'
                )
            if isinstance(self.seed, (int, np.integer)):
                if self.seed < 0 or self.seed > 2**32 - 1:
                    raise ValueError(
                        'Argument "seed" must be an integer between 0 and '
                        f'2**32-1, found {self.seed}.'
                    )
            else:
                if self.chain_ids is None:
                    raise ValueError(
                        'List of per-chain seeds cannot be evaluated without '
                        'corresponding list of chain_ids.'
                    )
                if len(self.seed) != len(self.chain_ids):
                    raise ValueError(
                        'Number of seeds must match number of chains, found '
                        f'{len(self.seed)} seeds for {len(self.chain_ids)} '
                        'chains.'
                    )
                for seed in self.seed:
                    if seed < 0 or seed > 2**32 - 1:
                        raise ValueError(
                            'Argument "seed" must be an integer value between '
                            f'0 and 2**32-1, found {seed}'
                        )

        if isinstance(self.data, str):
            if not os.path.exists(self.data):
                raise ValueError(f'no such file {self.data}')
        elif self.data is not None and not isinstance(self.data, (str, dict)):
            raise ValueError('Argument "data" must be string or dict')

        if self.inits is not None:
            if isinstance(self.inits, (float, int, np.floating, np.integer)):
                if self.inits < 0:
                    raise ValueError(
                        f'Argument "inits" must be > 0, found {self.inits}'
                    )
            elif isinstance(self.inits, str):
                if not os.path.exists(self.inits):
                    raise ValueError(f'no such file {self.inits}')
            elif isinstance(self.inits, list):
                if self.chain_ids is None:
                    raise ValueError(
                        'List of inits files cannot be evaluated without '
                        'corresponding list of chain_ids.'
                    )

                if len(self.inits) != len(self.chain_ids):
                    raise ValueError(
                        'Number of inits files must match number of chains, '
                        f'found {len(self.inits)} inits files for '
                        f'{len(self.chain_ids)} chains.'
                    )
                for inits in self.inits:
                    if not os.path.exists(inits):
                        raise ValueError(f'no such file {inits}')

    def compose_command(
        self,
        idx: int,
        csv_file: str,
        *,
        diagnostic_file: Optional[str] = None,
        profile_file: Optional[str] = None,
        num_chains: Optional[int] = None
    ) -> List[str]:
        """
        Compose CmdStan command for non-default arguments.
        """
        cmd: List[str] = []
        if idx is not None and self.chain_ids is not None:
            if idx < 0 or idx > len(self.chain_ids) - 1:
                raise ValueError(
                    f'index ({idx}) exceeds number of chains '
                    f'({len(self.chain_ids)})'
                )
            cmd.append(self.model_exe)  # type: ignore # guaranteed by validate
            cmd.append(f'id={self.chain_ids[idx]}')
        else:
            cmd.append(self.model_exe)  # type: ignore # guaranteed by validate

        if self.seed is not None:
            if not isinstance(self.seed, list):
                cmd.append('random')
                cmd.append(f'seed={self.seed}')
            else:
                cmd.append('random')
                cmd.append(f'seed={self.seed[idx]}')
        if self.data is not None:
            cmd.append('data')
            cmd.append(f'file={self.data}')
        if self.inits is not None:
            if not isinstance(self.inits, list):
                cmd.append(f'init={self.inits}')
            else:
                cmd.append(f'init={self.inits[idx]}')
        cmd.append('output')
        cmd.append(f'file={csv_file}')
        if diagnostic_file:
            cmd.append(f'diagnostic_file={diagnostic_file}')
        if profile_file:
            cmd.append(f'profile_file={profile_file}')
        if self.refresh is not None:
            cmd.append(f'refresh={self.refresh}')
        if self.sig_figs is not None:
            cmd.append(f'sig_figs={self.sig_figs}')
        cmd = self.method_args.compose(idx, cmd)
        if num_chains:
            cmd.append(f'num_chains={num_chains}')
        return cmd
