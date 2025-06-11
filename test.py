import torch
from rl4co.envs.graph.wal.env import WALEnv
from rl4co.models.zoo.am import AttentionModel
from rl4co.utils.trainer import RL4COTrainer
import pandas as pd
import folium
from collections import Counter
import numpy as np

def test_trained_model_fast():
    # load Walmart data
    walmart_df = pd.read_csv("./wal_envdata/walmart730.csv")
    # NY_block_df = pd.read_csv("./wal_envdata/NY_block_without_longisland.csv")
    print(f"the size of Walmart data set: {len(walmart_df)} locations")
    
    sample_size = len(walmart_df)
    
    # create test environment
    walmart_env = WALEnv(generator_params=dict(
        num_loc=sample_size, 
        to_choose=90,  # choose 8 locations
        memory_efficient="ultra"
    ))
    device = torch.device("cpu")
    print(f"using device: {device}")
    
    # create a new model instance, but with small test_data_size
    print("creating model instance (small data set for fast test)...")
    model_for_loading = AttentionModel(
        walmart_env,
        baseline="rollout",
        train_data_size=100,    # small data set
        val_data_size=50,       # small data set  
        test_data_size=20,     # 🔥 set small test data set!
        optimizer_kwargs={'lr': 1e-4}
    )
    
    # load weights from checkpoint
    print("loading model weights from checkpoint...")
    # load state dict instead of the whole model
    checkpoint = torch.load("checkpoints/download-v3/last-v1.ckpt", map_location=device, weights_only=False)
    model_for_loading.load_state_dict(checkpoint['state_dict'], strict=False)
    model_for_loading = model_for_loading.to(device)
    model_for_loading.eval()
    print("✅ model weights loaded successfully!")
    # create test data
    walmart_test_batch = walmart_env.reset(batch_size=[10],phase="test").to(device)
    
    with torch.no_grad():
        walmart_output = model_for_loading.policy(
            walmart_test_batch, walmart_env,
            phase="test", decode_type="sampling"
        )
    
    print(f"Walmart real data ({sample_size} locations) average reward: {walmart_output['reward'].mean().item():.4f}")
    
    # find the maximum reward and its corresponding action
    max_reward_value, max_reward_idx = torch.max(walmart_output['reward'], dim=0)
    max_reward_action = walmart_output['actions'][max_reward_idx]
    
    print(f"max sampling reward: {max_reward_value.item():.4f}")
    print(f"index of max sampling reward: {max_reward_idx.item()}")
    print(f"max sampling action sequence: {max_reward_action}")
    print(f"shape of max sampling action sequence: {max_reward_action.shape}")
    
    # convert action sequence to numpy array for processing
    action_sequence = max_reward_action.cpu().numpy()
    print(f"action sequence: {action_sequence}")
    
    # count the number of times each location appears
    action_counts = Counter(action_sequence)
    print(f"count of each location: {action_counts}")
    
    # create map visualization
    print("\n🗺️ creating map visualization...")
    
    # calculate the center of the map
    map_center = [walmart_df['lat'].mean(), walmart_df['lon'].mean()]
    
    # create a basic map
    m = folium.Map(
        location=map_center, 
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # add all walmart locations (gray small points)
    for idx, row in walmart_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            color='gray',
            fillColor='lightgray',
            fillOpacity=0.6,
            popup=f"location {idx}: ({row['lat']:.4f}, {row['lon']:.4f})"
        ).add_to(m)
    
    # add the selected locations (using different colors based on the number of times they appear)
    for location_idx, count in action_counts.items():
        if location_idx < len(walmart_df):  # ensure the index is valid
            lat = walmart_df.iloc[location_idx]['lat']
            lon = walmart_df.iloc[location_idx]['lon']
            
            # determine the color and size based on the number of times they appear
            if count == 1:
                color = 'yellow'
                fill_color = 'yellow'
                radius = 8
                popup_text = f"selected once - location {location_idx}: ({lat:.4f}, {lon:.4f})"
            elif count == 2:
                color = 'red'
                fill_color = 'red'
                radius = 10
                popup_text = f"selected twice - location {location_idx}: ({lat:.4f}, {lon:.4f})"
            else:
                color = 'purple'
                fill_color = 'purple'
                radius = 12
                popup_text = f"selected {count} times - location {location_idx}: ({lat:.4f}, {lon:.4f})"
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fillColor=fill_color,
                fillOpacity=0.8,
                weight=3,
                popup=popup_text
            ).add_to(m)
    
    # add legend
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>Graph Legend</h4>
    <p><i class="fa fa-circle" style="color:gray"></i> all Walmart locations candidates</p>
    <p><i class="fa fa-circle" style="color:yellow"></i> regular store</p>
    <p><i class="fa fa-circle" style="color:red"></i> food store</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # save map
    map_filename = "walmart_locations_map.html"
    m.save(map_filename)
    print(f"✅ map saved as: {map_filename}")
    
    # output the detailed information of the selected locations
    print(f"\n📍 the detailed information of the selected locations:")
    for location_idx, count in sorted(action_counts.items()):
        if location_idx < len(walmart_df):
            lat = walmart_df.iloc[location_idx]['lat']
            lon = walmart_df.iloc[location_idx]['lon']
            print(f"  location {location_idx}: ({lat:.6f}, {lon:.6f}) - appears {count} times")

if __name__ == "__main__":
    test_trained_model_fast() 