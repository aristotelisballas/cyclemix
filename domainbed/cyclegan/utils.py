def get_sources(dataset: str, targets: list):
    if dataset == 'PACS':
        # ENVIRONMENTS = ["A", "C", "P", "S"]
        domains = ["art_painting", "cartoon", "photo", "sketch"]
        domain_names = {
            0: "art_painting",
            1: "cartoon",
            2: "photo",
            3: "sketch"
        }
        for i in targets:
            domains.remove(domain_names[i])
        return domains

    else:
        raise NotImplementedError("Dataset not implemented yet for GANs")
