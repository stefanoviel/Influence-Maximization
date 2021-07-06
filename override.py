import copy
import collections
from inspyred.ec import *

def new_evolve_ea(self,pop_size=100, seeds=None, maximize=True, bounder=None, **args):
        """Perform the evolution.
        
        This function creates a population and then runs it through a series
        of evolutionary epochs until the terminator is satisfied. The general
        outline of an epoch is selection, variation, evaluation, replacement,
        migration, archival, and observation. The function returns a list of
        elements of type ``Individual`` representing the individuals contained
        in the final population.
        
        Arguments:
        
        - *generator* -- the function to be used to generate candidate solutions 
        - *evaluator* -- the function to be used to evaluate candidate solutions
        - *pop_size* -- the number of Individuals in the population (default 100)
        - *seeds* -- an iterable collection of candidate solutions to include
          in the initial population (default None)
        - *maximize* -- Boolean value stating use of maximization (default True)
        - *bounder* -- a function used to bound candidate solutions (default None)
        - *args* -- a dictionary of keyword arguments

        The *bounder* parameter, if left as ``None``, will be initialized to a
        default ``Bounder`` object that performs no bounding on candidates.
        Note that the *_kwargs* class variable will be initialized to the *args* 
        parameter here. It will also be modified to include the following 'built-in' 
        keyword argument:
        
        - *_ec* -- the evolutionary computation (this object)
        
        """
        self._kwargs = args
        self._kwargs['_ec'] = self
        self._kwargs["generations"] = 0
        if seeds is None:
            seeds = []
        if bounder is None:
            bounder = Bounder()
        
        self.termination_cause = None
        self.bounder = bounder
        self.maximize = maximize
        self.population = []
        self.archive = []
        generator= self._kwargs["generator"]
        evaluator = self._kwargs["evaluator"]
        # Create the initial population.
        if not isinstance(seeds, collections.Sequence):
            seeds = [seeds]
        initial_cs = copy.copy(seeds)
        num_generated = max(pop_size - len(seeds), 0)
        i = 0
        self.logger.debug('generating initial population')
        while i < num_generated:
            cs = generator(random=self._random, args=self._kwargs)
            initial_cs.append(cs)
            i += 1
        self.logger.debug('evaluating initial population')
        initial_fit =evaluator(candidates=initial_cs, args=self._kwargs)
        #print('Variators{0}'.format(self.variator))
        for cs, fit in zip(initial_cs, initial_fit):
            if fit is not None:
                ind = Individual(cs, maximize=maximize)
                ind.fitness = fit
                self.population.append(ind)
            else:
                self.logger.warning('excluding candidate {0} because fitness received as None'.format(cs))
        self.logger.debug('population size is now {0}'.format(len(self.population)))
        
        self.num_evaluations = len(initial_fit)
        self.num_generations = 0
        self._kwargs["generations"] = self.num_generations
        
        self.logger.debug('archiving initial population')
        self.archive = self.archiver(random=self._random, population=list(self.population), archive=list(self.archive), args=self._kwargs)
        self.logger.debug('archive size is now {0}'.format(len(self.archive)))
        self.logger.debug('population size is now {0}'.format(len(self.population)))
                
        if isinstance(self.observer, collections.Iterable):
            for obs in self.observer:
                self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(obs.__name__, self.num_generations, self.num_evaluations))
                obs(archiver=list(self.archive), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)
        else:
            self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(self.observer.__name__, self.num_generations, self.num_evaluations))
            self.observer(archiver=list(self.archive), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)
        
        while not self._should_terminate(list(self.population), self.num_generations, self.num_evaluations):
            # Select individuals.
            self.logger.debug('selection using {0} at generation {1} and evaluation {2}'.format(self.selector.__name__, self.num_generations, self.num_evaluations))
            parents = self.selector(random=self._random, population=list(self.population), args=self._kwargs)
            self.logger.debug('selected {0} candidates'.format(len(parents)))
            parent_cs = [copy.deepcopy(i.candidate) for i in parents]
            offspring_cs = parent_cs
            
            if isinstance(self.variator, collections.Iterable):
                for op in self.variator:
                    self.logger.debug('variation using {0} at generation {1} and evaluation {2}'.format(op.__name__, self.num_generations, self.num_evaluations))
                    #print('variation using {0} at generation {1} and evaluation {2}'.format(op.__name__, self.num_generations, self.num_evaluations))

                    offspring_cs = op(random=self._random, candidates=offspring_cs, args=self._kwargs)
            else:
                self.logger.debug('variation using {0} at generation {1} and evaluation {2}'.format(self.variator.__name__, self.num_generations, self.num_evaluations))
                offspring_cs = self.variator(random=self._random, candidates=offspring_cs, args=self._kwargs)
            self.logger.debug('created {0} offspring'.format(len(offspring_cs)))
            
            # Evaluate offspring.
            self.logger.debug('evaluation using {0} at generation {1} and evaluation {2}'.format(evaluator.__name__, self.num_generations, self.num_evaluations))
            self._kwargs["generations"] = self.num_generations

            offspring_fit = evaluator(candidates=offspring_cs, args=self._kwargs)
            offspring = []
            for cs, fit in zip(offspring_cs, offspring_fit):
                if fit is not None:
                    off = Individual(cs, maximize=maximize)
                    off.fitness = fit
                    offspring.append(off)
                else:
                    self.logger.warning('excluding candidate {0} because fitness received as None'.format(cs))
            self.num_evaluations += len(offspring_fit)        

            # Replace individuals.
            self.logger.debug('replacement using {0} at generation {1} and evaluation {2}'.format(self.replacer.__name__, self.num_generations, self.num_evaluations))
            self.population = self.replacer(random=self._random, population=self.population, parents=parents, offspring=offspring, args=self._kwargs)
            self.logger.debug('population size is now {0}'.format(len(self.population)))
            # Migrate individuals.
            self.logger.debug('migration using {0} at generation {1} and evaluation {2}'.format(self.migrator.__name__, self.num_generations, self.num_evaluations))
            self.population = self.migrator(random=self._random, population=self.population, args=self._kwargs)
            self.logger.debug('population size is now {0}'.format(len(self.population)))
            
            # Archive individuals.
            self.logger.debug('archival using {0} at generation {1} and evaluation {2}'.format(self.archiver.__name__, self.num_generations, self.num_evaluations))
            self.archive = self.archiver(random=self._random, archive=self.archive, population=list(self.population), args=self._kwargs)
            self.logger.debug('archive size is now {0}'.format(len(self.archive)))
            self.logger.debug('population size is now {0}'.format(len(self.population)))
          
            self.num_generations += 1
            if isinstance(self.observer, collections.Iterable):
                for obs in self.observer:
                    self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(obs.__name__, self.num_generations, self.num_evaluations))
                    obs(archiver=list(self.archive), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)
            else:
                self.logger.debug('observation using {0} at generation {1} and evaluation {2}'.format(self.observer.__name__, self.num_generations, self.num_evaluations))
                self.observer(archiver=list(self.archive), num_generations=self.num_generations, num_evaluations=self.num_evaluations, args=self._kwargs)
        return self.population



def new_evolve_nsga2(self, pop_size=100, seeds=None, maximize=True, bounder=None, **args):
        args.setdefault('num_selected', pop_size)
        args.setdefault('tournament_size', 5)
        return EvolutionaryComputation.new_evolve_ea(self, pop_size, seeds, maximize, bounder, **args)

