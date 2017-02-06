def remove_nodes(nodes_dict):
    keys_to_remove = []
    for k,v in nodes_dict.iteritems():
          if len(v) < 2:
              keys_to_remove.append(k)

    for key in set(keys_to_remove):
        del nodes_dict[key]
    return nodes_dict

def remove_weak_connections(nodes_dict,threshold):
    for k,v in nodes_dict.iteritems():
        nodes_dict[k] = [country for country in v if v.count(country) > threshold]

    #nodes_dict = {k: v for k, v in nodes_dict.iteritems() if v}
    return nodes_dict

def create_node_dict(df):
    nodes_dict = defaultdict(list)
    for items in df['countries_in_title']: #countries in every article
        if len(items)>1:
            for i, word in enumerate(items): # country for specific article
                nodes_dict[word] += (items[:i] + items[i+1:])

    #nodes_dict = remove_nodes(nodes_dict)
    nodes_dict = remove_weak_connections(nodes_dict,3)
    G=nx.Graph(nodes_dict)
    return G

def create_node_graph(df,graph_name):
    df= add_title_countries_to_df(df)
    nx.write_gml(create_node_dict(df), graph_name)
