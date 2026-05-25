from gymnasium.envs.registration import register

register(id="cookingEnv-v1",
         entry_point="src.environments.cooking_zoo.impl.environment:GymCookingEnvironment")
register(id="cookingEnvMA-v1",
         entry_point="src.environments.cooking_zoo.impl.environment:GymCookingEnvironmentMA")
register(id="cookingZooEnv-v0",
         entry_point="src.environments.cooking_zoo.impl.environment:CookingZooEnvironment")
