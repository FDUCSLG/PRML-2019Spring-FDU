import numpy as np

class SGD(object):
    def __init__(self, params, config):
        self.params = params
        self.config = self.init_config(config)
        self.optim_configs = {}
        for p in self.params:
            d = {k: v for k, v in config.items()}
            self.optim_configs[p] = d

    def init_config(self, config):
        config['learning_rate'] = config.get('learning_rate', 1e-2)
        return config

    def update(self, w, dw, config):
        w -= config['learning_rate'] * dw
        return w, config

    def step(self, grads):
        for p, w in self.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update(w, dw, config)
            self.params[p] = next_w
            self.optim_configs[p] = next_config
        return self.params


class Adam(object):
    def __init__(self, params, config):
        self.params = params
        self.config = self.init_config(config)
        self.optim_configs = {}
        for p in self.params:
            d = {k: v for k, v in config.items()}
            self.optim_configs[p] = d

    def init_config(self, config):
        config['learning_rate'] = config.get('learning_rate', 1e-3)
        config['beta1']         = config.get('beta1', 0.9)
        config['beta2']         = config.get('beta2', 0.999)
        config['epsilon']       = config.get('epsilon', 1e-8)
        config['m']             = config.get('m', None)
        config['v']             = config.get('v', None)
        config['t']             = config.get('t', 0)
        return config
    
    def update(self, x, dx, config):
        if config['m'] is None: config['m'] = np.zeros_like(x)
        if config['v'] is None: config['v'] = np.zeros_like(x)
        
        next_x = None
        beta1, beta2, eps = config['beta1'], config['beta2'], config['epsilon']
        t, m, v = config['t'], config['m'], config['v']
        m = beta1 * m + (1 - beta1) * dx
        v = beta2 * v + (1 - beta2) * (dx * dx)
        t += 1
        alpha = config['learning_rate'] * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        x -= alpha * (m / (np.sqrt(v) + eps))
        config['t'] = t
        config['m'] = m
        config['v'] = v
        next_x = x
        return next_x, config

    def step(self, grads):
        for p, w in self.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update(w, dw, config)
            self.params[p] = next_w
            self.optim_configs[p] = next_config
        return self.params

class Momentum(object):
    def __init__(self, params, config):
        self.params = params
        self.config = self.init_config(config)
        self.optim_configs = {}
        for p in self.params:
            d = {k: v for k, v in config.items()}
            self.optim_configs[p] = d

    def init_config(self, config):
        config['learning_rate'] = config.get('learning_rate', 1e-3)
        config['momentum']      = config.get('momentum', 0)
        config['m']             = config.get('m', None)
        return config

    def update(self, w, dw, config):
        if config['m'] is None: config['m'] = np.zeros_like(w)

        config['m'] = config['momentum'] * config['m'] + dw
        w -= config['learning_rate'] * config['m']
        return w, config

    def step(self, grads):
        for p, w in self.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update(w, dw, config)
            self.params[p] = next_w
            self.optim_configs[p] = next_config
        return self.params

class Nesterov(object):
    def __init__(self, params, config):
        self.params = params
        self.config = self.init_config(config)
        self.optim_configs = {}
        for p in self.params:
            d = {k: v for k, v in config.items()}
            self.optim_configs[p] = d

    def init_config(self, config):
        config['learning_rate'] = config.get('learning_rate', 1e-3)
        config['momentum']      = config.get('momentum', 0)
        config['m']             = config.get('m', None)
        return config

    # new version
    def update(self, w, dw, config):
        if config['m'] is None: config['m'] = np.zeros_like(w)

        config['m'] = config['momentum'] * config['m'] + dw
        w = w + (1 - config['momentum'] - config['learning_rate']) * config['m'] - dw
        return w, config

    def step(self, grads):
        for p, w in self.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update(w, dw, config)
            self.params[p] = next_w
            self.optim_configs[p] = next_config
        return self.params

class Adagrad(object):
    def __init__(self, params, config):
        self.params = params
        self.config = self.init_config(config)
        self.optim_configs = {}
        for p in self.params:
            d = {k: v for k, v in config.items()}
            self.optim_configs[p] = d

    def init_config(self, config):
        config['learning_rate'] = config.get('learning_rate', 1e-3)
        config['epsilon']       = config.get('epsilon', 1e-6)
        config['m']             = config.get('m', None)
        return config

    def update(self, w, dw, config):
        if config['m'] is None: config['m'] = np.zeros_like(w)

        config['m'] = config['m'] + dw ** 2
        w -= config['learning_rate'] * dw / np.sqrt(config['m'] + config['epsilon'])
        return w, config

    def step(self, grads):
        for p, w in self.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update(w, dw, config)
            self.params[p] = next_w
            self.optim_configs[p] = next_config
        return self.params

class Adadelta(object):
    def __init__(self, params, config):
        self.params = params
        self.config = self.init_config(config)
        self.optim_configs = {}
        for p in self.params:
            d = {k: v for k, v in config.items()}
            self.optim_configs[p] = d

    def init_config(self, config):
        config['learning_rate'] = config.get('learning_rate', 1e-3)
        config['epsilon']       = config.get('epsilon', 1e-6)
        config['rho']           = config.get('rho', 0.9)
        config['m']             = config.get('m', None)
        return config

    def update(self, w, dw, config):
        if config['m'] is None: config['m'] = np.zeros_like(w)
        
        rho, m = config['rho'], config['m']
        m = rho * m + (1 - rho) * dw ** 2
        w -= config['learning_rate'] * dw / np.sqrt(m + config['epsilon'])
        config['m'] = m
        return w, config

    def step(self, grads):
        for p, w in self.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update(w, dw, config)
            self.params[p] = next_w
            self.optim_configs[p] = next_config
        return self.params

class RMSProp(object):
    def __init__(self, params, config):
        self.params = params
        self.config = self.init_config(config)
        self.optim_configs = {}
        for p in self.params:
            d = {k: v for k, v in config.items()}
            self.optim_configs[p] = d

    def init_config(self, config):
        config['learning_rate'] = config.get('learning_rate', 1e-3)
        config['epsilon']       = config.get('epsilon', 1e-6)
        config['alpha']         = config.get('alpha', 0.99)
        config['m']             = config.get('m', None)
        return config

    def update(self, w, dw, config):
        if config['m'] is None: config['m'] = np.zeros_like(w)
        
        m, alpha = config['m'], config['alpha']
        m = alpha * m + (1 - alpha) * np.sqrt(dw)
        w -= config['learning_rate'] * dw / (np.sqrt(m) + config['epsilon'])
        config['m'] = m
        return w, config

    def step(self, grads):
        for p, w in self.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update(w, dw, config)
            self.params[p] = next_w
            self.optim_configs[p] = next_config
        return self.params

class Adamax(object):
    def __init__(self, params, config):
        self.params = params
        self.config = self.init_config(config)
        self.optim_configs = {}
        for p in self.params:
            d = {k: v for k, v in config.items()}
            self.optim_configs[p] = d

    def init_config(self, config):
        config['learning_rate'] = config.get('learning_rate', 1e-3)
        config['beta1']         = config.get('beta1', 0.9)
        config['beta2']         = config.get('beta2', 0.999)
        config['epsilon']       = config.get('epsilon', 1e-8)
        config['m']             = config.get('m', None)
        config['v']             = config.get('v', None)
        config['t']             = config.get('t', 0)
        return config

    def update(self, w, dw, config):
        if config['m'] is None: config['m'] = np.zeros_like(w)
        if config['v'] is None: config['v'] = np.zeros_like(w)
        # mu beta1, v beta2
        beta1, beta2 = config['beta1'], config['beta2']
        t, m, v = config['t'], config['m'], config['v']
        t += 1
        m = beta1 * m + (1 - beta1) * dw
        m = m / (1 - beta1 ** t)
        v = np.max(beta2 * v, np.abs(dw))
        w = w - config['learning_rate'] * m / (v + config['epsilon'])
        return w, config

    def step(self, grads):
        for p, w in self.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update(w, dw, config)
            self.params[p] = next_w
            self.optim_configs[p] = next_config
        return self.params

class Nadam(object):
    def __init__(self, params, config):
        self.params = params
        self.config = self.init_config(config)
        self.optim_configs = {}
        for p in self.params:
            d = {k: v for k, v in config.items()}
            self.optim_configs[p] = d

    def init_config(self, config):
        config['learning_rate']     = config.get('learning_rate', 1e-3)
        config['beta1']             = config.get('beta1', 0.9)
        config['beta2']             = config.get('beta2', 0.999)
        config['schedule_decay']    = config.get('schedule_decay', 4e-3)
        config['amsgrad']           = config.get('amsgrad', False)
        config['epsilon']           = config.get('epsilon', 1e-8)
        config['m']                 = config.get('m', None)
        config['v']                 = config.get('v', None)
        config['t']                 = config.get('t', 0)
        config['m_schedule']        = config.get('m_schedule', None)
        config['max_v']             = config.get('max_v', None)
        return config

    def update(self, w, dw, config):
        if config['m'] is None: config['m'] = np.zeros_like(w)
        if config['v'] is None: config['v'] = np.zeros_like(w)
        if config['m_schedule'] is None: config['m_schedule'] = 1
        if config['max_v'] is None: config['max_v'] = np.zeros_like(w)
        
        beta1, beta2 = config['beta1'], config['beta2']
        t, m, v = config['t'], config['m'], config['v']
        m_schedule = config['m_schedule']
        max_v = config['max_v']

        t += 1
        mom_cahce_t = beta1 * (1 - 0.5 * np.power(0.96, config['t'] * config['schedule_decay']))
        mom_cahce_t1 = beta1 * (1 - 0.5 * np.power(0.96, (config['t'] + 1) * config['schedule_decay']))
        m_schedule = m_schedule * mom_cahce_t

        m = m * beta1 + (1 - beta1) * dw
        m_t_prime = m / (1 - m_schedule * mom_cahce_t1)

        g_prime = dw / (1 - m_schedule)
        m_t_bar = (1 - mom_cahce_t) * g_prime + mom_cahce_t1 * m_t_prime

        v = v * beta2 + (1 - beta2) * (dw ** 2)
        if config['amsgrad']:
            max_v = np.max(max_v, v)
            v_t_prime = max_v / (1 - beta2 ** t)
        else:
            v_t_prime = v / (1 - beta2 ** t)
        denom = np.sqrt(v_t_prime) + config['epsilon']
        w -= config['learning_rate'] * m_t_bar / denom

        config['t'], config['m'], config['v'] = t, m, v
        config['m_schedule'] = m_schedule
        config['max_v'] = max_v

        return w, config

    def step(self, grads):
        for p, w in self.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update(w, dw, config)
            self.params[p] = next_w
            self.optim_configs[p] = next_config
        return self.params


class AdaBound(object):
    def __init__(self, params, config):
        self.params = params
        self.config = self.init_config(config)
        self.optim_configs = {}
        for p in self.params:
            d = {k: v for k, v in config.items()}
            self.optim_configs[p] = d

    def init_config(self, config):
        config['learning_rate'] = config.get('learning_rate', 1e-3)
        config['beta1']         = config.get('beta1', 0.9)
        config['beta2']         = config.get('beta2', 0.999)
        config['gamma']         = config.get('gamma', 0.1)
        config['final_lr']      = config.get('final_lr', 1e-3)
        config['amsgrad']       = config.get('amsgrad', False)
        config['epsilon']       = config.get('epsilon', 1e-8)
        config['m']             = config.get('m', None)
        config['v']             = config.get('v', None)
        config['t']             = config.get('t', 0)
        config['max_v']         = config.get('max_v', None)
        return config

    def update(self, w, dw, config):
        if config['m'] is None: config['m'] = np.zeros_like(w)
        if config['v'] is None: config['v'] = np.zeros_like(w)
        if config['max_v'] is None: config['max_v'] = np.zeros_like(w)
        
        max_v = config['max_v']
        beta1, beta2 = config['beta1'], config['beta2']
        t, m, v = config['t'], config['m'], config['v']
        
        t += 1 
        m = m * beta1 + (1 - beta1) * dw
        v = v * beta2 + (1 - beta2) * (dw ** 2)
        if config['amsbound']:
            max_v = np.max(max_v, v)
            denom = np.sqrt(max_v) + config['epsilon']
        else:
            denom = np.sqrt(v)

        bias_c1 = 1 - beta1 ** t
        bias_c2 = 1 - beta2 ** t

        step_size = config['learning_rate'] * np.sqrt(bias_c2) / bias_c1

        lb = config['final_lr'] * (1 - 1 / (config['gamma'] * t + 1))
        ub = config['final_lr'] * (1 + 1 / (config['gamma'] * t))
        step_size = np.full_like(denom, step_size)
        step_size = np.clamp(step_size / denom, lb, ub) * m
        w -= step_size

        config['m'] = m
        config['v'] = v
        config['t'] = t
        config['max_v'] = max_v

        return w, config

    def step(self, grads):
        for p, w in self.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update(w, dw, config)
            self.params[p] = next_w
            self.optim_configs[p] = next_config
        return self.params
