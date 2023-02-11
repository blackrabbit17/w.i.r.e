import ulid


def lexographical_id():
    """
     01AN4Z07BY      79KA1307SR9X4MV3
    |----------|    |----------------|
     Timestamp          Randomness
       48bits             80bits

    We can discard the random component and use the timestamp
    to generate a unique id for each run. This is at 1 second resolution.

    ulid doesn't expose a public interface for this functionality
    hence this utility function

    """
    return ulid.new().str[0:8]
