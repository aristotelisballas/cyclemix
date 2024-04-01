def get_sources(dataset: str, targets: list):
    if dataset == 'PACS':
        # ENVIRONMENTS = ["A", "C", "P", "S"]
        domains = ["art_painting", "cartoon", "photo", "sketch"]
        for i in targets:
            domains.remove(domains[i])
        return domains

    else:
        raise NotImplementedError("Dataset not implemented yet for GANs")
