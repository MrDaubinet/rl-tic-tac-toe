<script>
  // Component
  import Marker from "./Marker.svelte"
  // Data prop
  let { state, selectMarker, cellValues = [] } = $props();
</script>

<div class="grid grid-cols-3 gap-0">
  {#each state as value, index}
    <div class="relative">
      <Marker 
        value={value}
        index={index}
        selectMarker={selectMarker}
      />

      {#if value === 0 && cellValues[index] !== null}
        <span class="absolute mr-3 text-xs text-gray-400 bottom-1 right-1 select-none pointer-events-none">
          {cellValues[index]?.toFixed(2)}
        </span>
      {/if}
    </div>
  {/each}
</div> 