"""
Topic Hierarchy Web UI
Displays nets, subnets, and segments in an expandable/collapsible interface
"""

from flask import Flask, render_template, jsonify
import json
from collections import defaultdict

app = Flask(__name__)


def load_hierarchy():
    """Build hierarchy containing ALL nets/subnets, overlaying segments from coded_responses.json.

    - Preloads full net/subnet list from data/topic_hierarchy.json so empty nets/subnets appear.
    - Adds segments from data/coded_responses.json into the preinitialized containers.
    - Falls back to creating entries for unknown net/subnet IDs found in segments.
    """
    # 1) Preload full hierarchy of nets and subnets (labels only)
    with open('data/topic_hierarchy.json', 'r', encoding='utf-8') as f:
        topic_meta = json.load(f)

    nets_meta = topic_meta.get('nets', {})

    hierarchy = defaultdict(lambda: {
        'net_name': '',
        'subnets': defaultdict(lambda: {
            'subnet_name': '',
            'segments': []
        }),
        'direct_segments': []
    })

    for net_id, net_info in nets_meta.items():
        # Initialize every net
        hierarchy[net_id]['net_name'] = net_info.get('label', net_id)

        # Initialize all declared subnets (if any)
        subnets_info = net_info.get('subnets', {}) or {}
        for subnet_id, subnet_info in subnets_info.items():
            hierarchy[net_id]['subnets'][subnet_id]['subnet_name'] = subnet_info.get('label', subnet_id)

    # 2) Overlay segments from coded_responses.json
    with open('data/coded_responses.json', 'r', encoding='utf-8') as f:
        coded = json.load(f)

    for response in coded:
        for segment in response.get('segments', []):
            net_id = segment.get('net_id', 'Unknown')
            net_name = segment.get('net', net_id)
            subnet_id = segment.get('subnet_id', 'Unknown')
            subnet_name = segment.get('subnet', subnet_id)

            # Ensure net exists (in case of unknown/new)
            if not hierarchy[net_id]['net_name']:
                hierarchy[net_id]['net_name'] = net_name or net_id

            segment_data = {
                'segment_id': segment.get('segment_id', ''),
                'text': segment.get('text', '')
            }

            # Treat Unknown or same-as-net as direct segments
            if subnet_id == 'Unknown' or subnet_id == net_id:
                hierarchy[net_id]['direct_segments'].append(segment_data)
            else:
                # Ensure subnet exists (for unknown/new)
                if not hierarchy[net_id]['subnets'][subnet_id]['subnet_name']:
                    hierarchy[net_id]['subnets'][subnet_id]['subnet_name'] = subnet_name or subnet_id
                hierarchy[net_id]['subnets'][subnet_id]['segments'].append(segment_data)

    # 3) Convert to list and compute counts; include ALL nets/subnets even if empty
    result = []

    # Sort nets by label for user-friendly display; fallback to id
    def net_sort_key(item):
        net_id, net_data = item
        return (net_data.get('net_name') or net_id or '').lower()

    for net_id, net_data in sorted(hierarchy.items(), key=net_sort_key):
        net_obj = {
            'net_id': net_id,
            'net_name': net_data['net_name'] or net_id,
            'subnets': [],
            'direct_segments': net_data['direct_segments'],
            'total_segments': len(net_data['direct_segments'])
        }

        # Sort subnets by label
        def subnet_sort_key(item):
            subnet_id, subnet_data = item
            return (subnet_data.get('subnet_name') or subnet_id or '').lower()

        for subnet_id, subnet_data in sorted(net_data['subnets'].items(), key=subnet_sort_key):
            subnet_obj = {
                'subnet_id': subnet_id,
                'subnet_name': subnet_data['subnet_name'] or subnet_id,
                'segments': subnet_data['segments'],
                'segment_count': len(subnet_data['segments'])
            }
            net_obj['subnets'].append(subnet_obj)
            net_obj['total_segments'] += len(subnet_data['segments'])

        result.append(net_obj)

    return result


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/api/hierarchy')
def get_hierarchy():
    """API endpoint to get the topic hierarchy"""
    hierarchy = load_hierarchy()
    return jsonify(hierarchy)


if __name__ == '__main__':
    print("Starting Topic Hierarchy Web UI...")
    print("Open your browser to: http://localhost:5000")
    app.run(debug=True, port=5000)

