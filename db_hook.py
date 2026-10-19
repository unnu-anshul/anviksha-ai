def save(report):
    import db
    db.init()
    return db.save_run(report)
