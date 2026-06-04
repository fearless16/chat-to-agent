"""AES-256-GCM encrypted credential storage using cryptography.fernet.Fernet."""

from __future__ import annotations

from cryptography.fernet import Fernet


class CredentialVault:
    """Encrypted, key-value credential store backed by Fernet (AES-256-GCM).

    Each value is encrypted individually with a master key. The vault supports
    export/import for migration and online key rotation.
    """

    def __init__(self, master_key: bytes | None = None) -> None:
        """Initialise the vault.

        Parameters
        ----------
        master_key:
            A 32-byte, URL-safe-base64-encoded Fernet key. If ``None`` a new
            key is generated via :meth:`generate_key`.
        """
        if master_key is None:
            master_key = self.generate_key()
        self._fernet = Fernet(master_key)
        self._store: dict[str, bytes] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, key: str, value: str) -> None:
        """Encrypt and store *value* under *key*.

        If *key* already exists the previous value is overwritten.
        """
        self._store[key] = self._fernet.encrypt(value.encode("utf-8"))

    def retrieve(self, key: str) -> str | None:
        """Decrypt and return the value for *key*, or ``None`` if missing."""
        ciphertext = self._store.get(key)
        if ciphertext is None:
            return None
        return self._fernet.decrypt(ciphertext).decode("utf-8")

    def delete(self, key: str) -> bool:
        """Remove *key* from the vault.  Returns ``True`` if the key existed."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def list_keys(self) -> list[str]:
        """Return a list of all stored keys (in insertion order)."""
        return list(self._store.keys())

    def export_encrypted(self) -> dict[str, bytes]:
        """Export all credentials as a *key* → ciphertext mapping.

        The returned dict is safe to serialise and transmit: values are still
        encrypted under the current master key.
        """
        return dict(self._store)

    def import_encrypted(self, data: dict[str, bytes]) -> None:
        """Merge previously exported encrypted data into this vault.

        Existing keys are preserved (they are **not** overwritten).
        """
        for key, ciphertext in data.items():
            if key not in self._store:
                self._store[key] = ciphertext

    def rotate_key(self, new_key: bytes) -> None:
        """Rotate the master key, re-encrypting all values in-place.

        Parameters
        ----------
        new_key:
            A 32-byte URL-safe-base64-encoded Fernet key.
        """
        # Decrypt everything with the old Fernet first.
        plaintexts: dict[str, str] = {}
        for key in list(self._store.keys()):
            plaintexts[key] = self.retrieve(key)  # type: ignore[assignment]

        # Swap in the new Fernet and re-encrypt.
        self._fernet = Fernet(new_key)
        self._store = {}
        for key, value in plaintexts.items():
            self._store[key] = self._fernet.encrypt(value.encode("utf-8"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def generate_key() -> bytes:
        """Generate a fresh 32-byte URL-safe-base64-encoded Fernet key."""
        return Fernet.generate_key()
