import re
import torch
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')


# content reward 测试
# completions = [
#     [{'content': 'The answer is 10.'}],
#     [{'content': 'The answer is 20.'}],
#     [{'content': 'The answer is 30.'}],
# ]
# responses = [
#     [{'content': 'The answer is 10.hi'}],
#     [{'content': 'The answer is 30.'}],
#     [{'content': 'The answer is 30.hi'}],
# ]
def content_reward(completions,sigmoid_scale=10.0, sigmoid_shift=0.7,
                       high_reward_baseline=1.0, low_reward_baseline=0.0, negative_reward_threshold=0.6, negative_reward_value=-0.1, **kwargs):
    embeddings = model.encode([completion[0]['content'] for completion in completions])
    # print("reward_len Received kwargs:", kwargs)  # 添加调试打印
    responses = kwargs['_response']
    embeddings_2 =  model.encode([completion[0]['content'] for completion in responses]) 
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings_2)
    rewards = []
    for i in range(len(completions)):
        similarity = similarity_matrix[i][i].item()
        print('similarity:',similarity)
        # if similarity > 0.9:
        #     rewards.append(1.0)
        # elif similarity < 0.6:
        #     rewards.append(0.0)
        # else:
        #     rewards.append(similarity-0.3)
        # 使用 Sigmoid 函数计算奖励值
        reward_sigmoid = torch.sigmoid(torch.tensor([(similarity - sigmoid_shift) * sigmoid_scale])).item()
        reward = reward_sigmoid * (high_reward_baseline - low_reward_baseline) + low_reward_baseline
        # 可选的负奖励
        if similarity < negative_reward_threshold:
            reward = negative_reward_value
        print('reward:',reward)
        rewards.append(reward)
    return rewards

# format reward 测试
# completions = [
#     [{'content': '*The answer is 10.*'}],
#     [{'content': '*The answer is 20.*You are ok'}],
#     [{'content': '*The answer is 30.*   '}],
# ]
def format_reward(completions, **kwargs):
    """Reward function that checks if:
    1. The content contains an even number of * markers
    2. Not all content is enclosed within * markers
    """
    def check_format(content):
        # Count total number of * markers
        star_count = content.count('*')
        if star_count == 0 or star_count % 2 != 0:
            return False
        # Calculate total length and content length within stars
        content = content.strip()
        total_length = len(content)
        star_content = 0
        in_star = False
        star_segment_length = 2  # Include the * markers themselves
        for char in content:
            if char == '*':
                if in_star:
                    star_content += star_segment_length
                    star_segment_length = 2
                in_star = not in_star
            elif in_star:
                star_segment_length += 1
        # Return True if there's content outside of stars
        return star_content < total_length
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [check_format(content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]