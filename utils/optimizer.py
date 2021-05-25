from torch.optim.lr_scheduler import LambdaLR, MultiStepLR


def get_lr_scheduler(cfg, optimizer, max_iter):
    assert max_iter > 0
    dtype, kwargs = cfg.dtype, cfg.kwargs
    if dtype == "Step":
        steps = [step * max_iter for step in kwargs['steps']]
        scheduler = MultiStepLR(optimizer, milestones=steps, gamma=kwargs['gamma'])
    elif dtype == "Poly":
        num_groups = len(optimizer.param_groups)
        def lambdaf(cur_iter):
            return (1 - (cur_iter * 1.0) / max_iter) ** kwargs['power']
        scheduler = LambdaLR(optimizer, lr_lambda=[lambdaf]*num_groups)
    else:
        raise NotImplementedError("not support lr policy")

    return scheduler
